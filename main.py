import asyncio
import logging
import time
import warnings
from dotenv import load_dotenv
from core.graph import build_financial_graph

load_dotenv()
warnings.filterwarnings("ignore", category=ResourceWarning)
# Silence the benign Google GenAI AFC info log
logging.getLogger("google_genai.models").setLevel(logging.ERROR)


def extract_text_from_chunk(content) -> str:
    """Extracts clean text from LangChain's structured Gemini content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


async def run_langgraph_pipeline(user_query: str):
    graph = build_financial_graph()

    start_time = time.perf_counter()
    ttft = None

    print(f"\n[LangGraph] Executing workflow for: '{user_query}'...\n" + "=" * 65)

    async for msg_chunk, metadata in graph.astream(
        {"user_query": user_query},
        stream_mode="messages",
    ):
        if metadata.get("langgraph_node") == "synthesizer":
            text_token = extract_text_from_chunk(msg_chunk.content)
            if text_token:
                if ttft is None:
                    ttft = time.perf_counter() - start_time
                print(text_token, end="", flush=True)

    total_time = time.perf_counter() - start_time
    print("\n" + "=" * 65)
    print(f"[Metrics] TTFT: {ttft:.3f}s | Total Graph Execution: {total_time:.3f}s\n")


if __name__ == "__main__":
    asyncio.run(run_langgraph_pipeline("Tata Motors"))