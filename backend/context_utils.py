from typing import Iterable, List

from backend.schemas import ContextItem

FOLLOW_UP_MARKERS = (
    "在哪",
    "哪里",
    "在哪儿",
    "怎么",
    "怎么办",
    "如何",
    "多少",
    "几点",
    "电话",
    "地址",
    "这个",
    "那个",
    "它",
    "他",
    "她",
    "呢",
    "吗",
)

PRICE_MARKERS = ("多少钱", "多少", "价格", "费用", "收费", "贵吗")
LOCATION_MARKERS = ("在哪", "哪里", "在哪儿", "地址", "位置")
METHOD_MARKERS = ("怎么", "怎么办", "如何", "咋办", "怎样")
CONTACT_MARKERS = ("电话", "联系方式", "联系", "怎么联系")


def compact_text(text: str, limit: int = 80) -> str:
    normalized = " ".join(text.strip().split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "..."


def get_recent_context_items(context: Iterable[ContextItem], limit: int = 6) -> List[ContextItem]:
    items = [item for item in context if item.content.strip()]
    return items[-limit:]


def is_follow_up_question(question: str) -> bool:
    text = "".join(question.strip().split())
    if not text:
        return False

    if len(text) <= 8:
        return True

    return any(marker in text for marker in FOLLOW_UP_MARKERS)


def find_last_user_topic(context_items: List[ContextItem]) -> str:
    for item in reversed(context_items):
        if item.source == "chat" and item.role == "user":
            return compact_text(item.content, 60)
    return ""


def format_non_chat_context(context_items: List[ContextItem]) -> str:
    parts = []
    for item in context_items:
        if item.source == "chat":
            continue
        label = item.metadata.get("label") or item.source
        parts.append(f"{label}: {compact_text(item.content, 80)}")
    return "；".join(parts[:3])


def build_retrieval_question(question: str, context: List[ContextItem]) -> str:
    recent_items = get_recent_context_items(context)
    if not recent_items:
        return question

    last_user_topic = find_last_user_topic(recent_items)
    extra_context = format_non_chat_context(recent_items)

    if not is_follow_up_question(question):
        if extra_context:
            return f"{question}。补充上下文：{extra_context}"
        return question

    if not last_user_topic:
        return question

    standalone_question = build_standalone_question(last_user_topic, question)
    if extra_context:
        return f"{standalone_question}。补充上下文：{extra_context}"
    return standalone_question


def build_standalone_question(topic: str, follow_up: str) -> str:
    topic = compact_text(topic, 40).rstrip("，。！？、 ")
    follow_up = "".join(follow_up.strip().split())
    if not topic or not follow_up:
        return follow_up or topic

    if topic in follow_up:
        return follow_up

    if any(marker in follow_up for marker in PRICE_MARKERS):
        return f"{topic}{follow_up if follow_up.endswith('？') else follow_up + '？'}"

    if any(marker in follow_up for marker in LOCATION_MARKERS):
        return f"{topic}{follow_up if follow_up.endswith('？') else follow_up + '？'}"

    if any(marker in follow_up for marker in METHOD_MARKERS):
        return f"{topic}{follow_up if follow_up.endswith('？') else follow_up + '？'}"

    if any(marker in follow_up for marker in CONTACT_MARKERS):
        return f"{topic}的{follow_up if follow_up.endswith('？') else follow_up + '？'}"

    if follow_up.endswith(("？", "?", "吗", "呢")):
        return f"{topic}，{follow_up}"

    return f"{topic}相关：{follow_up}"
