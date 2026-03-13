import os
from http import HTTPStatus
from typing import List

import dashscope
from dashscope import Generation
from dotenv import load_dotenv

from backend.schemas import KnowledgeItem, Citation

load_dotenv()

API_KEY = os.getenv("DASHSCOPE_API_KEY")
MODEL_NAME = os.getenv("QWEN_MODEL", "qwen-plus")


def build_evidence_text(hits: List[KnowledgeItem]) -> str:
    # 这里把命中的 KnowledgeItem 展开成统一证据文本，供大模型一次性阅读。
    # 这一步不做摘要压缩，尽量保留原始证据，减少模型误解。
    blocks = []

    for item in hits:
        block = (
            f"经验贴ID: {item.id}\n"
            f"标题: {item.title}\n"
            f"内容: {item.content}\n"
            f"标签: {', '.join(item.tags)}"
        )
        blocks.append(block)

    return "\n\n---\n\n".join(blocks)


def generate_answer(question: str, hits: List[KnowledgeItem]):
    # 1) 没有命中证据，不调用模型
    if not hits:
        return {
            "answer": "暂时没有检索到直接相关的经验贴，建议补充更具体的问题。",
            "citations": []
        }

    # 2) API Key 校验
    if not API_KEY:
        return {
            "answer": "生成层未配置 DASHSCOPE_API_KEY，暂时返回规则回答。",
            "citations": [Citation(id=item.id, title=item.title) for item in hits]
        }

    evidence_text = build_evidence_text(hits)

    # 当前 prompt 设计强调“只能基于证据回答”，避免模型凭空补充物业规则。
    # 同时尽量减少裸 Markdown 输出，让前端展示更自然。
    system_prompt = (
        "你是一个物业客服知识助手。\n"
        "你的任务是：基于给定的经验贴证据，回答业主问题。\n"
        "要求：\n"
        "1. 只能依据经验贴内容回答，不要编造。\n"
        "2. 语言简洁、自然，像真实物业客服。\n"
        "3. 如果经验贴证据不足，要明确说信息不足。\n"
        "4. 优先整合最相关的经验贴内容。\n"
        "5. 默认输出自然中文，不要主动输出 Markdown 标记符号（如 **、#）。"
    )

    user_prompt = (
        f"用户问题：{question}\n\n"
        f"经验贴证据：\n{evidence_text}\n\n"
        "请直接输出一段适合回复给业主的中文答案。"
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
            answer_text = response.output.choices[0].message.content.strip()
        else:
            # 模型调用失败时，给一个可见的降级结果，避免直接 500
            answer_text = (
                f"模型调用失败，已返回规则回答。"
                f"错误码: {response.code}，错误信息: {response.message}"
            )

    except Exception as e:
        answer_text = f"模型调用异常，已返回规则回答。异常信息: {str(e)}"

    # citations 仍然保留给接口和日志使用，哪怕当前前端不展示。
    citations = [Citation(id=item.id, title=item.title) for item in hits]

    # 兜底：如果模型没返回有效文本，就回退到最相关知识的规则式回答。
    if not answer_text.strip():
        top = hits[0]
        answer_text = f"根据已有经验贴，优先参考《{top.title}》：{top.content}"

    return {
        "answer": answer_text,
        "citations": citations
    }
