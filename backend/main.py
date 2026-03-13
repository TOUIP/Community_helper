from pathlib import Path
import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.logging_utils import get_logger, get_runs_log_path
from backend.query_rewriter import rewrite_question_with_context
from backend.schemas import AskRequest, AskResponse
from backend.retriever import retrieve
from backend.generator import generate_answer


BASE_DIR = Path(__file__).resolve().parent.parent

FRONTEND_DIR = BASE_DIR / "frontend"
RUNS_PATH = get_runs_log_path()
app_logger = get_logger("community_helper.app", "app.log")

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
    # 检索和生成保持解耦：retrieve 只负责找证据，generate_answer 只负责组织回答。
    retrieval_question = rewrite_question_with_context(req.question, req.context)
    if retrieval_question != req.question:
        app_logger.info(
            "rewrite original=%r retrieval=%r context_items=%s",
            req.question,
            retrieval_question,
            len(req.context),
        )
    else:
        app_logger.info("question=%r context_items=%s", req.question, len(req.context))

    hits = retrieve(retrieval_question, top_k=3)
    # 生成层也使用“改写后的完整问题”，这样回答不会再围着“在哪里/多少钱”这类模糊追问打转。
    generated = generate_answer(retrieval_question, hits)

    result = AskResponse(
        question=req.question,
        hits=hits,
        answer=generated["answer"],
        citations=generated["citations"]
    )

    log_payload = result.model_dump()
    if retrieval_question != req.question:
        log_payload["retrieval_question"] = retrieval_question
        log_payload["context"] = [item.model_dump() for item in req.context]

    with RUNS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_payload, ensure_ascii=False) + "\n")

    return result


# 根路径直接挂载 frontend 目录，因此部署时不需要额外的静态文件服务。
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
