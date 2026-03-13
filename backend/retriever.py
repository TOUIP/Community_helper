import json
import re
from pathlib import Path
from typing import Iterable, List

from sentence_transformers import SentenceTransformer

from backend.logging_utils import get_logger
from backend.schemas import KnowledgeItem

KB_DIR = Path(__file__).parent / "kb"
INDEX_PATH = Path(__file__).parent.parent / "vector_index.json"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
VECTOR_WEIGHT = 0.72
LEXICAL_WEIGHT = 0.28

# 在线检索与离线建索引共用同一个 embedding 模型，保证向量空间一致。
model = SentenceTransformer(MODEL_NAME)
_retrieval_mode_logged = False
retrieval_logger = get_logger("community_helper.retriever", "retriever.log")


def load_kb() -> List[KnowledgeItem]:
    all_items = []

    for file_path in KB_DIR.glob("*.json"):
        data = json.loads(file_path.read_text(encoding="utf-8"))
        items = [KnowledgeItem(**item) for item in data]
        all_items.extend(items)

    return all_items


def load_vector_index() -> List[dict]:
    # 当前线上检索优先使用本地向量索引；它是由 backend.vector_index 预先生成的产物。
    if not INDEX_PATH.exists():
        return []

    data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_query_terms(text: str) -> List[str]:
    """
    轻量关键词切分：
    - 英文/数字按连续串保留
    - 中文按连续片段切成双字词，同时保留短片段本身
    这样不引入额外分词依赖，也能比逐字匹配稳一些。
    """
    normalized = normalize_text(text)
    terms = []
    seen = set()

    for token in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", normalized):
        candidates: List[str] = []
        if re.fullmatch(r"[a-z0-9]+", token):
            candidates.append(token)
        else:
            if len(token) <= 2:
                candidates.append(token)
            else:
                candidates.extend(token[i:i + 2] for i in range(len(token) - 1))
                candidates.append(token)

        for item in candidates:
            if item and item not in seen:
                seen.add(item)
                terms.append(item)

    return terms


def ratio(part: Iterable[str], whole: Iterable[str]) -> float:
    part_set = set(part)
    whole_set = set(whole)
    if not part_set or not whole_set:
        return 0.0
    return len(part_set & whole_set) / len(part_set)


def lexical_match_score(question: str, title: str, content: str, tags: List[str]) -> float:
    query = normalize_text(question)
    title_text = normalize_text(title)
    content_text = normalize_text(content)
    tags_text = normalize_text(" ".join(tags))
    terms = extract_query_terms(query)

    if not query:
        return 0.0

    score = 0.0

    # 整句命中优先级最高，尤其适合“物业费怎么交”“卫生服务站在哪”这类短问题。
    if query in title_text:
        score += 3.2
    if query in tags_text:
        score += 2.8
    if query in content_text:
        score += 2.2

    for term in terms:
        if term in title_text:
            score += 1.8
        if term in tags_text:
            score += 1.5
        if term in content_text:
            score += 0.9

    # 用字符覆盖率补足中文短问句，避免只靠向量时出现“语义像但业务词不对”。
    query_chars = [ch for ch in query if not ch.isspace()]
    score += 1.6 * ratio(query_chars, title_text + tags_text)
    score += 0.8 * ratio(query_chars, content_text)

    # 归一化到 0-1，便于与向量分数做加权融合。
    max_score = max(len(terms) * 4.2 + 4.5, 1.0)
    return min(score / max_score, 1.0)


def build_item_from_record(record: dict) -> KnowledgeItem:
    return KnowledgeItem(
        id=str(record.get("id", "")),
        title=str(record.get("title", "")),
        content=str(record.get("content", "")),
        tags=record.get("tags") or [],
    )


def log_retrieval_mode(mode: str, extra: str = "") -> None:
    global _retrieval_mode_logged
    if _retrieval_mode_logged:
        return

    suffix = f" {extra}" if extra else ""
    retrieval_logger.info("mode=%s%s", mode, suffix)
    _retrieval_mode_logged = True


def retrieve(question: str, top_k: int = 3) -> List[KnowledgeItem]:
    vector_index = load_vector_index()

    if vector_index:
        log_retrieval_mode("vector_index", f"path={INDEX_PATH} count={len(vector_index)}")
        # 只要索引文件存在，就走“向量 + 关键词”的混合召回。
        query_embedding = model.encode(question).tolist()
        scored = []

        for record in vector_index:
            item = build_item_from_record(record)
            dense_score = (cosine_similarity(query_embedding, record.get("embedding", [])) + 1) / 2
            lexical_score = lexical_match_score(question, item.title, item.content, item.tags)
            final_score = VECTOR_WEIGHT * dense_score + LEXICAL_WEIGHT * lexical_score
            scored.append((final_score, lexical_score, dense_score, item))

        scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        return [item for score, _, _, item in scored if score > 0.18][:top_k]

    # 这个分支是历史 fallback：在没有 vector_index.json 时，仍然能退回本地字符匹配。
    log_retrieval_mode("fallback_kb", f"path={KB_DIR}")
    kb = load_kb()

    scored = []
    for item in kb:
        score = lexical_match_score(question, item.title, item.content, item.tags)
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for score, item in scored if score > 0][:top_k]
