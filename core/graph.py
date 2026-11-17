import time
from typing import TypedDict, Dict, Any, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from core.router import resolve_ticker
from agents.quant_agent import fetch_quantitative_metrics
from agents.news_agent import fetch_news_sentiment
from agents.context_agent import fetch_company_fundamentals
from core.synthesizer import SYSTEM_INSTRUCTION

load_dotenv()

# Initialize once at module load to reuse HTTP/TLS connections across runs
SYNTHESIZER_LLM = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.3,
    thinking_level="minimal",  # Eliminates ~10-20s hidden reasoning delay
)


class MarketIntelligenceState(TypedDict):
    user_query: str
    ticker: str
    company_name: str
    wiki_query: str
    exchange: str
    is_indian_stock: bool
    error: str
    metrics: Dict[str, Any]
    news_headlines: List[Dict[str, str]]
    business_overview: str
    final_report: str


async def router_node(state: MarketIntelligenceState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    resolved = await resolve_ticker(state["user_query"], prefer_india=True)
    if "error" in resolved:
        return {"error": resolved["error"]}
    print(f"[Node: router] {resolved['ticker']} resolved in {time.perf_counter() - t0:.2f}s")
    return {
        "ticker": resolved["ticker"],
        "company_name": resolved["company_name"],
        "wiki_query": resolved["wiki_query"],
        "exchange": resolved["exchange"],
        "is_indian_stock": resolved["is_indian_stock"],
    }


async def quant_node(state: MarketIntelligenceState) -> Dict[str, Any]:
    if state.get("error"):
        return {}
    data = await fetch_quantitative_metrics(state["ticker"])
    return {"metrics": data}


async def news_node(state: MarketIntelligenceState) -> Dict[str, Any]:
    if state.get("error"):
        return {}
    data = await fetch_news_sentiment(
        state["wiki_query"], is_indian_stock=state["is_indian_stock"]
    )
    return {"news_headlines": data}


async def context_node(state: MarketIntelligenceState) -> Dict[str, Any]:
    if state.get("error"):
        return {}
    summary = await fetch_company_fundamentals(state["wiki_query"])
    return {"business_overview": summary}


async def synthesizer_node(state: MarketIntelligenceState) -> Dict[str, Any]:
    if state.get("error"):
        return {"final_report": f"Error: {state['error']}"}

    payload = (
        f"Company: {state['company_name']} ({state['ticker']} | {state['exchange']})\n"
        f"Metrics: {state['metrics']}\n"
        f"News Headlines: {state['news_headlines']}\n"
        f"Business Overview: {state['business_overview']}"
    )

    response = await SYNTHESIZER_LLM.ainvoke([
        SystemMessage(content=SYSTEM_INSTRUCTION),
        HumanMessage(content=payload)
    ])
    return {"final_report": response.content}


def build_financial_graph():
    builder = StateGraph(MarketIntelligenceState)

    builder.add_node("router", router_node)
    builder.add_node("quant_agent", quant_node)
    builder.add_node("news_agent", news_node)
    builder.add_node("context_agent", context_node)
    builder.add_node("synthesizer", synthesizer_node)

    builder.add_edge(START, "router")

    # Fan-Out
    builder.add_edge("router", "quant_agent")
    builder.add_edge("router", "news_agent")
    builder.add_edge("router", "context_agent")

    # Fan-In
    builder.add_edge("quant_agent", "synthesizer")
    builder.add_edge("news_agent", "synthesizer")
    builder.add_edge("context_agent", "synthesizer")

    builder.add_edge("synthesizer", END)

    return builder.compile()