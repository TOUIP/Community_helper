from typing import Any, Dict, List

from pydantic import BaseModel, Field

class ContextItem(BaseModel):
    role: str = Field(..., description="上下文角色，如 user / assistant / system")
    content: str = Field(..., description="上下文正文")
    source: str = Field(default="chat", description="上下文来源，如 chat / memory / profile")
    kind: str = Field(default="message", description="上下文类型，如 message / note / fact")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="预留给未来扩展的元数据")


# 这些 schema 是整个项目的“共享数据契约”，前后端和检索/生成链路都依赖它们。
class AskRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    # context 保持为结构化列表，而不是单一 history 字符串，
    # 这样后续可以逐步接入聊天记录、用户画像、业务上下文等多种信息源。
    context: List[ContextItem] = Field(default_factory=list, description="可选上下文")

# KnowledgeItem 是检索层的标准中间结构：
# 不管知识来自本地 JSON 还是 Supabase，都会先映射成这个形状。
class KnowledgeItem(BaseModel):
    id: str
    title: str
    content: str
    tags: List[str] = []

class Citation(BaseModel):
    id: str
    title: str

# AskResponse 仍然保留 hits / citations，虽然当前前端不展示它们，
# 这样可以避免后续调试、日志分析和接口兼容性受到影响。
class AskResponse(BaseModel):
    question: str
    hits: List[KnowledgeItem]
    answer: str
    citations: List[Citation]
