import json
from pathlib import Path
from typing import List
from backend.schemas import KnowledgeItem

KB_DIR = Path(__file__).parent / "kb"


def load_kb() -> List[KnowledgeItem]:
    all_items = []

    for file_path in KB_DIR.glob("*.json"):
        data = json.loads(file_path.read_text(encoding="utf-8"))
        items = [KnowledgeItem(**item) for item in data]
        all_items.extend(items)

    return all_items


def retrieve(question: str, top_k: int = 3) -> List[KnowledgeItem]:
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