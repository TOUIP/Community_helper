from pathlib import Path
import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.schemas import AskRequest, AskResponse
from backend.retriever import retrieve
from backend.generator import generate_answer


BASE_DIR = Path(__file__).resolve().parent.parent

FRONTEND_DIR = BASE_DIR / "frontend"

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

RUNS_PATH = LOG_DIR / "runs.jsonl"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"ok": True, "msg": "Service is running."}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    hits = retrieve(req.question, top_k=3)
    generated = generate_answer(req.question, hits)

    result = AskResponse(
        question=req.question,
        hits=hits,
        answer=generated["answer"],
        citations=generated["citations"]
    )

    with RUNS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result.model_dump(), ensure_ascii=False) + "\n")

    return result


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")