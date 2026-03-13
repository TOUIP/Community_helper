import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from supabase import Client, create_client

from backend.schemas import KnowledgeItem

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "posts").strip() or "posts"

# 阶段一只做数据源替换。publish_status / verified_status 的真实枚举值
# 以你们 Supabase 实际定义为准，这里用环境变量承接，后续可直接调整。
PUBLISHED_STATUS_VALUE = os.getenv("SUPABASE_PUBLISHED_STATUS", "published").strip()
VERIFIED_STATUS_VALUE = os.getenv("SUPABASE_VERIFIED_STATUS", "").strip()
PAGE_SIZE = int(os.getenv("SUPABASE_PAGE_SIZE", "500"))


def get_supabase_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("缺少 Supabase 配置，请检查 SUPABASE_URL / SUPABASE_KEY")

    return create_client(SUPABASE_URL, SUPABASE_KEY)


def _extract_text_values(value: Any) -> List[str]:
    """尽量把 tags 的 jsonb 值提取成可读文本列表。"""
    results: List[str] = []

    if value is None:
        return results

    if isinstance(value, str):
        text = value.strip()
        if text:
            results.append(text)
        return results

    if isinstance(value, (int, float, bool)):
        results.append(str(value))
        return results

    if isinstance(value, list):
        for item in value:
            results.extend(_extract_text_values(item))
        return results

    if isinstance(value, dict):
        preferred_keys = ["name", "label", "title", "text", "value", "tag"]
        for key in preferred_keys:
            if key in value:
                results.extend(_extract_text_values(value.get(key)))

        if not results:
            for nested_value in value.values():
                results.extend(_extract_text_values(nested_value))

        return results

    return results


def _normalize_tags(tags_value: Any) -> List[str]:
    values = _extract_text_values(tags_value)

    normalized: List[str] = []
    seen = set()

    for item in values:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)

    return normalized


def fetch_posts_from_supabase() -> List[Dict[str, Any]]:
    client = get_supabase_client()

    all_rows: List[Dict[str, Any]] = []
    start = 0

    while True:
        # 这里只做阶段一需要的最小筛选：
        # 1. publish_status 必须匹配
        # 2. verified_status 过滤是可选的
        # 3. title 为空的记录不进入知识库
        response = (
            client.table(SUPABASE_TABLE)
            .select(
                "id, created_at, title, description, images, tags, "
                "activity_start_date, activity_end_date, user_id, "
                "publish_status, verified_status, community_id, verified_time"
            )
            .eq("publish_status", PUBLISHED_STATUS_VALUE)
            .range(start, start + PAGE_SIZE - 1)
            .execute()
        )

        rows = response.data or []
        if not rows:
            break

        if VERIFIED_STATUS_VALUE:
            rows = [
                row
                for row in rows
                if str(row.get("verified_status") or "").strip() == VERIFIED_STATUS_VALUE
            ]

        rows = [row for row in rows if str(row.get("title") or "").strip()]
        all_rows.extend(rows)

        if len(response.data or []) < PAGE_SIZE:
            break

        start += PAGE_SIZE

    return all_rows


def map_post_to_knowledge_item(post: Dict[str, Any]) -> Optional[KnowledgeItem]:
    title = str(post.get("title") or "").strip()
    if not title:
        return None

    # 本阶段只做字段映射，不额外拼接 images，也不做摘要改写。
    content = str(post.get("description") or "").strip()
    tags = _normalize_tags(post.get("tags"))

    return KnowledgeItem(
        id=str(post.get("id")),
        title=title,
        content=content,
        tags=tags,
    )


def load_kb_from_supabase() -> List[KnowledgeItem]:
    posts = fetch_posts_from_supabase()

    items: List[KnowledgeItem] = []
    for post in posts:
        item = map_post_to_knowledge_item(post)
        if item is not None:
            items.append(item)

    return items
