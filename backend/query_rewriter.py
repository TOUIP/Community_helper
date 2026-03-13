import os
from http import HTTPStatus
from typing import List

from dashscope import Generation
from dotenv import load_dotenv

from backend.context_utils import build_retrieval_question, get_recent_context_items
from backend.logging_utils import get_logger
from backend.schemas import ContextItem

load_dotenv()

API_KEY = os.getenv("DASHSCOPE_API_KEY")
MODEL_NAME = os.getenv("QWEN_MODEL", "qwen-plus")
rewriter_logger = get_logger("community_helper.rewriter", "rewriter.log")


def format_context_for_rewrite(context: List[ContextItem]) -> str:
    # 统一把最近上下文整理成稳定文本块，给 LLM 做 Query Rewrite。
    lines = []
    for item in get_recent_context_items(context, limit=6):
        role = item.role
        source = item.source
        content = " ".join(item.content.strip().split())
        if not content:
            continue
        lines.append(f"[{source}/{role}] {content}")
    return "\n".join(lines)


def should_try_llm_rewrite(question: str, context: List[ContextItem]) -> bool:
    if not context:
        return False
    text = "".join(question.strip().split())
    if not text:
        return False
    # 这里只在“像追问”的场景下才额外调用模型，控制成本和延迟。
    return len(text) <= 12 or any(token in text for token in ("哪", "怎么", "多少", "电话", "地址", "它", "这个", "那个"))


def rewrite_question_with_context(question: str, context: List[ContextItem]) -> str:
    # 先生成规则版 fallback，确保就算 LLM 不可用也不会失去多轮能力。
    rule_based = build_retrieval_question(question, context)

    if not should_try_llm_rewrite(question, context) or not API_KEY:
        return rule_based

    context_text = format_context_for_rewrite(context)
    if not context_text:
        return rule_based

    # 这个模型只负责“把追问改成适合检索的独立问题”，不负责回答。
    system_prompt = (
        "你是一个检索前的问题改写助手。\n"
        "你的任务是：结合最近上下文，把用户当前追问改写成一个适合知识检索的、完整明确的中文问题。\n"
        "要求：\n"
        "1. 只输出改写后的单句问题，不要解释。\n"
        "2. 如果当前问题本身已经完整明确，就原样输出。\n"
        "3. 不要编造上下文里没有的信息。\n"
        "4. 输出要自然，像真实用户会问的问题。"
    )

    user_prompt = (
        f"最近上下文：\n{context_text}\n\n"
        f"当前问题：{question}\n\n"
        "请输出改写后的完整问题。"
    )

    try:
        response = Generation.call(
            api_key=API_KEY,
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            result_format="message",
            enable_thinking=False,
        )

        if response.status_code == HTTPStatus.OK:
            rewritten = response.output.choices[0].message.content.strip()
            if rewritten:
                rewriter_logger.info("llm_rewrite original=%r rewritten=%r", question, rewritten)
                return rewritten

        rewriter_logger.warning(
            "llm_rewrite_failed_status original=%r code=%s message=%s",
            question,
            getattr(response, "code", ""),
            getattr(response, "message", ""),
        )
    except Exception as exc:
        rewriter_logger.warning("llm_rewrite_exception original=%r error=%s", question, exc)

    return rule_based
