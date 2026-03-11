import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from backend.retriever import load_kb

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
INDEX_PATH = Path(__file__).parent / "vector_index.json"

model = SentenceTransformer(MODEL_NAME)


def build_index():
    kb = load_kb()

    records = []
    for item in kb:
        text = f"{item.title} {item.content} {' '.join(item.tags)}"
        embedding = model.encode(text).tolist()

        records.append({
            "id": item.id,
            "title": item.title,
            "content": item.content,
            "tags": item.tags,
            "embedding": embedding
        })

    INDEX_PATH.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"向量索引已生成：{INDEX_PATH}")
    print(f"共写入 {len(records)} 条记录")


if __name__ == "__main__":
    build_index()