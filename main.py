import asyncio
import logging
import time
import warnings
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore", category=ResourceWarning)
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


async def analyze_query(graph, user_query: str):
    start_time = time.perf_counter()
    ttft = None

    print("\n" + "=" * 70)

    async for msg_chunk, metadata in graph.astream(
        {"user_query": user_query, "dossiers": []},
        stream_mode="messages",
    ):
        if metadata.get("langgraph_node") == "synthesizer":
            text_token = extract_text_from_chunk(msg_chunk.content)
            if text_token:
                if ttft is None:
                    ttft = time.perf_counter() - start_time
                print(text_token, end="", flush=True)

    total_time = time.perf_counter() - start_time
    print("\n" + "=" * 70)
    ttft_str = f"{ttft:.2f}s" if ttft else "N/A"
    print(f"[Metrics] TTFT: {ttft_str} | End-to-End: {total_time:.2f}s")


async def cli_loop():
    print("\n======================================================================")
    print("  NeuroPulse Market Agent (LangGraph Map-Reduce CLI)")
    print("  Try: 'Reliance' | 'Compare Tata Motors vs M&M' | 'Why is Zomato moving?'")
    print("  Type 'exit' to quit")
    print("======================================================================")

    # Build graph after banner prints so CLI feels instant
    from core.graph import build_financial_graph
    graph = build_financial_graph()

    while True:
        try:
            user_input = await asyncio.to_thread(input, "\nQuery > ")
            query = user_input.strip()

            if not query:
                continue
            if query.lower() in {"exit", "quit", "q"}:
                print("Goodbye!")
                break

            await analyze_query(graph, query)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break


if __name__ == "__main__":
    asyncio.run(cli_loop())