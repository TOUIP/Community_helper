import json
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from backend.retriever import load_kb
from backend.supabase_kb import load_kb_from_supabase

load_dotenv()

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
INDEX_PATH = Path(__file__).parent.parent / "vector_index.json"

# 这个模型用于“离线建索引”，必须和在线 retriever 使用的模型一致。
model = SentenceTransformer(MODEL_NAME)


def get_knowledge_items():
    """
    阶段一默认优先使用 Supabase 数据源。
    如果 Supabase 当前不可用，则回退到本地 kb，避免索引构建完全中断。
    """
    try:
        kb = load_kb_from_supabase()
        if kb:
            print(f"已从 Supabase 加载 {len(kb)} 条知识")
            return kb

        print("Supabase 未返回可用知识，回退到本地 kb")
    except Exception as exc:
        print(f"从 Supabase 加载知识失败，回退到本地 kb。原因: {exc}")

    kb = load_kb()
    print(f"已从本地 kb 加载 {len(kb)} 条知识")
    return kb


def build_index():
    kb = get_knowledge_items()

    records = []
    for item in kb:
        # 阶段一仍然是一篇 post 对应一个向量，不做 chunking。
        text = f"{item.title} {item.content} {' '.join(item.tags)}"
        embedding = model.encode(text).tolist()

        records.append({
            "id": item.id,
            "title": item.title,
            "content": item.content,
            "tags": item.tags,
            "embedding": embedding
        })

    temp_path = INDEX_PATH.with_suffix(".json.tmp")
    temp_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    temp_path.replace(INDEX_PATH)

    print(f"向量索引已生成：{INDEX_PATH}")
    print(f"共写入 {len(records)} 条记录")


if __name__ == "__main__":
    build_index()
