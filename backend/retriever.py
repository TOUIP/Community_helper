import json
from pathlib import Path
from typing import List

from sentence_transformers import SentenceTransformer

from backend.schemas import KnowledgeItem

KB_DIR = Path(__file__).parent / "kb"
INDEX_PATH = Path(__file__).parent.parent / "vector_index.json"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# 在线检索与离线建索引共用同一个 embedding 模型，保证向量空间一致。
model = SentenceTransformer(MODEL_NAME)


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


def retrieve(question: str, top_k: int = 3) -> List[KnowledgeItem]:
    vector_index = load_vector_index()

    if vector_index:
        # 只要索引文件存在，就走真正的向量召回。
        query_embedding = model.encode(question).tolist()
        scored = []

        for record in vector_index:
            score = cosine_similarity(query_embedding, record.get("embedding", []))
            item = KnowledgeItem(
                id=str(record.get("id", "")),
                title=str(record.get("title", "")),
                content=str(record.get("content", "")),
                tags=record.get("tags") or [],
            )
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for score, item in scored if score > 0][:top_k]

    # 这个分支是历史 fallback：在没有 vector_index.json 时，仍然能退回本地字符匹配。
    kb = load_kb()

    scored = []
    for item in kb:
        text = f"{item.title} {item.content} {' '.join(item.tags)}"
        score = 0

        for ch in question:
            if ch.strip() and ch in text:
                score += 1

        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for score, item in scored if score > 0][:top_k]
