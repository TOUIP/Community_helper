###定义“系统里所有关键数据长什么样”
from pydantic import BaseModel, Field
from typing import List

#定义用户请求 /ask 接口时，传入的数据长什么样
class AskRequest(BaseModel):
    question: str = Field(..., description="用户问题")

#定义经验贴的标准结构
class KnowledgeItem(BaseModel):
    id: str
    title: str
    content: str
    tags: List[str] = []

#定义引用信息的数据结构
class Citation(BaseModel):
    id: str
    title: str

#定义/ask 接口最终返回的数据长什么样
class AskResponse(BaseModel):
    question: str
    hits: List[KnowledgeItem]
    answer: str
    citations: List[Citation]