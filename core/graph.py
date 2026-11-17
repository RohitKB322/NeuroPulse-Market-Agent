import asyncio
import json
import operator
import time
from typing import TypedDict, Dict, Any, List, Optional, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver

from core.router import plan_query_execution, resolve_ticker
from agents.quant_agent import fetch_quantitative_metrics
from agents.news_agent import fetch_news_sentiment
from agents.context_agent import fetch_company_fundamentals

load_dotenv()

SINGLE_BRIEF_PROMPT = """You are an institutional-grade Equity Research & Market Sentiment Analyst.
Synthesize a crisp, high-signal market intelligence brief from the provided company dossier:
1. **Snapshot & Price Action**: Current price, daily % change, 52-week range position, and formatted Market Cap (₹ Lakh Crore / Crore for INR; $B / $T for USD).
2. **News & Catalyst Sentiment**: Overall short-term sentiment (Bullish / Bearish / Neutral / Mixed) with a 1-line rationale, plus 2–3 bullet points highlighting key news catalysts.
3. **Business Fundamentals**: 2 concise sentences on core revenue drivers (skip if omitted for a news-only query).
4. **Analyst Takeaway**: One sharp, objective takeaway on valuation/momentum context.

Keep it scannable, data-dense, and under 250 words."""

COMPARISON_PROMPT = """You are a senior Buy-Side Equity Strategist comparing multiple companies side by side.
Using the provided company dossiers, produce a sharp comparative intelligence report:
1. **Head-to-Head Metrics Matrix**: A clean Markdown table comparing Ticker, Price (with daily % change), Market Cap, 52W Range Position, and Short-Term News Sentiment.
2. **Catalyst & Momentum Divergence**: 3–4 concise bullet points contrasting what is driving each stock right now.
3. **Relative Verdict**: A clear, objective conclusion on which company currently holds stronger momentum vs. better valuation cushion (tailor to focus_aspect if provided).

Keep it crisp, analytical, and under 350 words."""

FOLLOW_UP_PROMPT = """You are a senior Equity Research Assistant answering a follow-up question in an ongoing market analysis chat.
Use the cached company dossiers and recent conversation history to answer directly, accurately, and concisely without repeating the entire initial report."""

_SYNTHESIZER_LLM = None


def _get_synthesizer_llm():
    global _SYNTHESIZER_LLM
    if _SYNTHESIZER_LLM is None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        primary = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            temperature=0.3,
            thinking_level="minimal",
            max_retries=1,
        )
        fallback = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            temperature=0.3,
            max_retries=1,
        )
        _SYNTHESIZER_LLM = primary.with_fallbacks([fallback])
    return _SYNTHESIZER_LLM


def merge_dossiers_reducer(existing: List[Dict[str, Any]], new_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merges new worker dossiers into state, updating existing tickers cleanly without duplicates."""
    if not existing:
        existing = []
    by_ticker = {d.get("ticker", d.get("query")): d for d in existing}
    for item in new_items:
        key = item.get("ticker", item.get("query"))
        by_ticker[key] = item
    return list(by_ticker.values())


class MarketIntelligenceState(TypedDict):
    user_query: str
    intent: str
    companies_to_fetch: List[str]
    need_fundamentals: bool
    focus_aspect: Optional[str]
    dossiers: Annotated[List[Dict[str, Any]], merge_dossiers_reducer]
    chat_history: Annotated[List[Dict[str, str]], operator.add]
    final_report: str


class WorkerState(TypedDict):
    company_query: str
    need_fundamentals: bool


async def supervisor_node(state: MarketIntelligenceState) -> Dict[str, Any]:
    existing_dossiers = state.get("dossiers", [])
    cached_names = [
        f"{d.get('company_name')} ({d.get('ticker')})"
        for d in existing_dossiers
        if "ticker" in d
    ]

    plan = await plan_query_execution(state["user_query"], cached_companies=cached_names)

    return {
        "intent": plan.intent,
        "companies_to_fetch": plan.companies_to_fetch,
        "need_fundamentals": plan.need_fundamentals,
        "focus_aspect": plan.focus_aspect,
        "chat_history": [{"role": "user", "content": state["user_query"]}],
    }


def dispatch_workers(state: MarketIntelligenceState) -> List[Send]:
    targets = state.get("companies_to_fetch", [])
    if not targets:
        return [Send("synthesizer", state)]

    return [
        Send(
            "research_worker",
            {
                "company_query": company,
                "need_fundamentals": state.get("need_fundamentals", True),
            },
        )
        for company in targets
    ]


async def research_worker_node(state: WorkerState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    query = state["company_query"]
    resolved = await resolve_ticker(query, prefer_india=True)

    if "error" in resolved:
        return {"dossiers": [{"query": query, "error": resolved["error"], "elapsed_ms": round((time.perf_counter() - t0) * 1000)}]}

    ticker = resolved["ticker"]
    wiki_query = resolved["wiki_query"]
    is_indian = resolved["is_indian_stock"]

    tasks = [
        fetch_quantitative_metrics(ticker),
        fetch_news_sentiment(wiki_query, is_indian_stock=is_indian),
    ]
    if state.get("need_fundamentals", True):
        tasks.append(fetch_company_fundamentals(wiki_query))

    results = await asyncio.gather(*tasks)
    quant_data = results[0]
    news_data = results[1]
    context_data = results[2] if len(results) > 2 else "Skipped (not needed for query)."

    elapsed_ms = round((time.perf_counter() - t0) * 1000)

    dossier = {
        "ticker": ticker,
        "company_name": resolved["company_name"],
        "exchange": resolved["exchange"],
        "metrics": quant_data,
        "news_headlines": news_data,
        "business_overview": context_data,
        "elapsed_ms": elapsed_ms,
    }
    return {"dossiers": [dossier]}


async def synthesizer_node(state: MarketIntelligenceState) -> Dict[str, Any]:
    from langchain_core.messages import SystemMessage, HumanMessage

    dossiers = state.get("dossiers", [])
    if not dossiers:
        msg = "Could not identify any valid company or ticker in your query."
        return {"final_report": msg, "chat_history": [{"role": "assistant", "content": msg}]}

    intent = state.get("intent", "single_analysis")
    focus = state.get("focus_aspect")

    if intent == "follow_up":
        sys_prompt = FOLLOW_UP_PROMPT
    elif intent == "comparison" and len(dossiers) > 1:
        sys_prompt = COMPARISON_PROMPT
    else:
        sys_prompt = SINGLE_BRIEF_PROMPT

    recent_turns = state.get("chat_history", [])[-6:]
    user_payload = (
        f"Recent Conversation:\n{json.dumps(recent_turns, indent=2)}\n\n"
        f"Current Query: {state['user_query']}\n"
        f"Focus Aspect: {focus or 'General Overview'}\n\n"
        f"Company Dossiers:\n{json.dumps(dossiers, indent=2)}"
    )

    llm = _get_synthesizer_llm()
    response = await llm.ainvoke([
        SystemMessage(content=sys_prompt),
        HumanMessage(content=user_payload),
    ])

    report_text = response.content if isinstance(response.content, str) else str(response.content)
    return {
        "final_report": report_text,
        "chat_history": [{"role": "assistant", "content": report_text}],
    }


def build_financial_graph():
    builder = StateGraph(MarketIntelligenceState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("research_worker", research_worker_node)
    builder.add_node("synthesizer", synthesizer_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", dispatch_workers, ["research_worker", "synthesizer"])
    builder.add_edge("research_worker", "synthesizer")
    builder.add_edge("synthesizer", END)

    # Enable in-memory session checkpointing for multi-turn follow-ups
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)