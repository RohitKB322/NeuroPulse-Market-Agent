import json
import logging
import time
import warnings
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.graph import build_financial_graph

load_dotenv()
warnings.filterwarnings("ignore", category=ResourceWarning)
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

app = FastAPI(title="NeuroPulse Market Agent")

# Serve static frontend files
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Compile LangGraph once at server startup so connections & MemorySaver persist
GRAPH = build_financial_graph()


class ChatRequest(BaseModel):
    query: str
    thread_id: str = "default-session"


def extract_text_from_chunk(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def sse_pack(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


async def stream_graph_events(query: str, thread_id: str) -> AsyncGenerator[str, None]:
    start_time = time.perf_counter()
    ttft = None
    config = {"configurable": {"thread_id": thread_id}}

    # Notify UI that Supervisor has started
    yield sse_pack({"type": "node_status", "node": "supervisor", "status": "running"})

    try:
        async for stream_type, payload in GRAPH.astream(
            {"user_query": query},
            config=config,
            stream_mode=["updates", "messages"],
        ):
            if stream_type == "updates":
                # payload is {node_name: node_output_dict}
                for node_name, output in payload.items():
                    elapsed_ms = round((time.perf_counter() - start_time) * 1000)

                    if node_name == "supervisor":
                        targets = output.get("companies_to_fetch", [])
                        yield sse_pack({
                            "type": "supervisor_done",
                            "intent": output.get("intent", "single_analysis"),
                            "targets": targets,
                            "need_fundamentals": output.get("need_fundamentals", True),
                            "elapsed_ms": elapsed_ms,
                        })

                    elif node_name == "research_worker":
                        new_dossiers = output.get("dossiers", [])
                        for d in new_dossiers:
                            yield sse_pack({
                                "type": "worker_done",
                                "dossier": d,
                            })

            elif stream_type == "messages":
                msg_chunk, metadata = payload
                if metadata.get("langgraph_node") == "synthesizer":
                    token = extract_text_from_chunk(msg_chunk.content)
                    if token:
                        if ttft is None:
                            ttft = round((time.perf_counter() - start_time) * 1000)
                            yield sse_pack({"type": "synthesizer_start", "ttft_ms": ttft})
                        yield sse_pack({"type": "token", "text": token})

        total_ms = round((time.perf_counter() - start_time) * 1000)
        yield sse_pack({"type": "done", "total_ms": total_ms, "ttft_ms": ttft})

    except Exception as e:
        yield sse_pack({"type": "error", "message": str(e)})


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    index_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    return StreamingResponse(
        stream_graph_events(req.query, req.thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)