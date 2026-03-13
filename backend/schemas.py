from pydantic import BaseModel, Field
from typing import List

# 这些 schema 是整个项目的“共享数据契约”，前后端和检索/生成链路都依赖它们。
class AskRequest(BaseModel):
    question: str = Field(..., description="用户问题")

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
