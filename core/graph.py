import asyncio
import json
import operator
import time
from typing import TypedDict, Dict, Any, List, Optional, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from core.router import plan_query_execution, resolve_ticker
from agents.quant_agent import fetch_quantitative_metrics
from agents.news_agent import fetch_news_sentiment
from agents.context_agent import fetch_company_fundamentals

load_dotenv()

SINGLE_BRIEF_PROMPT = """You are an institutional-grade Equity Research & Market Sentiment Analyst.
Synthesize a crisp, high-signal market intelligence brief from the provided company dossier:
1. **Snapshot & Price Action**: Current price, daily % change, 52-week range position, P/E ratio (if available), and formatted Market Cap (₹ Lakh Crore / Crore for INR; $B / $T for USD).
2. **News & Catalyst Sentiment**: Overall short-term sentiment (Bullish / Bearish / Neutral / Mixed) with a 1-line rationale, plus 2–3 bullet points highlighting key news catalysts.
3. **Business Fundamentals**: 2 concise sentences on core revenue drivers (skip this section if business_overview is omitted for a news-only query).
4. **Analyst Takeaway**: One sharp, objective takeaway on valuation/momentum context.

Keep it scannable, data-dense, and under 250 words."""

COMPARISON_PROMPT = """You are a senior Buy-Side Equity Strategist comparing multiple companies side by side.
Using the provided company dossiers, produce a sharp comparative intelligence report:
1. **Head-to-Head Metrics Matrix**: A clean Markdown table comparing Ticker, Price (with daily % change), Market Cap, Trailing P/E, 52W Range Position, and Short-Term News Sentiment.
2. **Catalyst & Momentum Divergence**: 3–4 concise bullet points contrasting what is driving each stock right now (earnings, sector tailwinds, headwinds).
3. **Relative Verdict**: A clear, objective conclusion on which company currently holds stronger momentum vs. better valuation cushion (tailor directly to the user's focus_aspect if provided).

Keep it crisp, analytical, and under 350 words."""

_SYNTHESIZER_LLM = None


def _get_synthesizer_llm():
    """Lazy-loads primary fast Gemini model with automatic fallback on 503/429 spikes."""
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
            model="gemini-2.5-flash",
            temperature=0.3,
            max_retries=1,
        )
        _SYNTHESIZER_LLM = primary.with_fallbacks([fallback])
    return _SYNTHESIZER_LLM


# 1. Main Graph State (with operator.add reducer for parallel worker dossiers)
class MarketIntelligenceState(TypedDict):
    user_query: str
    intent: str
    companies: List[str]
    need_fundamentals: bool
    focus_aspect: Optional[str]
    dossiers: Annotated[List[Dict[str, Any]], operator.add]
    final_report: str


# 2. Worker Input State (passed via Send())
class WorkerState(TypedDict):
    company_query: str
    need_fundamentals: bool


# Node 1: Intelligent LLM Supervisor
async def supervisor_node(state: MarketIntelligenceState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    plan = await plan_query_execution(state["user_query"])
    elapsed = time.perf_counter() - t0
    print(
        f"[Supervisor] Intent: '{plan.intent}' | Targets: {plan.companies} | "
        f"Wiki: {plan.need_fundamentals} ({elapsed:.2f}s)"
    )
    return {
        "intent": plan.intent,
        "companies": plan.companies,
        "need_fundamentals": plan.need_fundamentals,
        "focus_aspect": plan.focus_aspect,
    }


# Conditional Edge: Dynamic Map Fan-Out via Send()
def dispatch_workers(state: MarketIntelligenceState) -> List[Send]:
    companies = state.get("companies", [])
    if not companies:
        return [Send("synthesizer", state)]

    return [
        Send(
            "research_worker",
            {
                "company_query": company,
                "need_fundamentals": state.get("need_fundamentals", True),
            },
        )
        for company in companies
    ]


# Node 2: Parallel Research Worker (runs once per target company simultaneously)
async def research_worker_node(state: WorkerState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    query = state["company_query"]
    resolved = await resolve_ticker(query, prefer_india=True)

    if "error" in resolved:
        return {"dossiers": [{"query": query, "error": resolved["error"]}]}

    ticker = resolved["ticker"]
    wiki_query = resolved["wiki_query"]
    is_indian = resolved["is_indian_stock"]

    # Run quant + news (+ optional wiki) concurrently inside this worker
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

    elapsed = time.perf_counter() - t0
    print(f"[Worker: {ticker}] Fetched all data in {elapsed:.2f}s")

    dossier = {
        "ticker": ticker,
        "company_name": resolved["company_name"],
        "exchange": resolved["exchange"],
        "metrics": quant_data,
        "news_headlines": news_data,
        "business_overview": context_data,
    }
    return {"dossiers": [dossier]}


# Node 3: Adaptive Streaming Synthesizer
async def synthesizer_node(state: MarketIntelligenceState) -> Dict[str, Any]:
    from langchain_core.messages import SystemMessage, HumanMessage

    dossiers = state.get("dossiers", [])
    if not dossiers:
        return {"final_report": "Could not identify any valid company or ticker in your query."}

    intent = state.get("intent", "single_analysis")
    focus = state.get("focus_aspect")
    sys_prompt = COMPARISON_PROMPT if (intent == "comparison" and len(dossiers) > 1) else SINGLE_BRIEF_PROMPT

    user_payload = (
        f"User Query: {state['user_query']}\n"
        f"Focus Aspect: {focus or 'General Overview'}\n\n"
        f"Collected Market Dossiers:\n{json.dumps(dossiers, indent=2)}"
    )

    llm = _get_synthesizer_llm()
    response = await llm.ainvoke([
        SystemMessage(content=sys_prompt),
        HumanMessage(content=user_payload),
    ])
    return {"final_report": response.content}


def build_financial_graph():
    builder = StateGraph(MarketIntelligenceState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("research_worker", research_worker_node)
    builder.add_node("synthesizer", synthesizer_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", dispatch_workers, ["research_worker", "synthesizer"])
    builder.add_edge("research_worker", "synthesizer")
    builder.add_edge("synthesizer", END)

    return builder.compile()