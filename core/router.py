import asyncio
from typing import Dict, Any, List, Literal, Optional
from pydantic import BaseModel, Field


class QueryPlan(BaseModel):
    intent: Literal["single_analysis", "comparison", "news_only"] = Field(
        description=(
            "'comparison' if comparing 2+ companies/stocks; "
            "'news_only' if asking only about recent news, catalysts, or why a stock moved today; "
            "'single_analysis' for general stock analysis or fundamentals."
        )
    )
    companies: List[str] = Field(
        description=(
            "List of normalized corporate names or tickers extracted from the query "
            "(e.g., ['Tata Motors', 'Mahindra & Mahindra'] or ['Reliance Industries'])."
        )
    )
    need_fundamentals: bool = Field(
        description="False if intent is 'news_only' or user only wants quick price/news; True otherwise."
    )
    focus_aspect: Optional[str] = Field(
        default=None,
        description="Specific angle requested by user (e.g., 'EV market share', 'valuation', 'earnings'), or None."
    )


_PLANNER_CHAIN = None


def _get_planner_chain():
    """Lazy-loads LangChain + Gemini structured planner so CLI startup is instant."""
    global _PLANNER_CHAIN
    if _PLANNER_CHAIN is None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            temperature=0.0,
            thinking_level="minimal",
            max_retries=1,
        )
        _PLANNER_CHAIN = llm.with_structured_output(QueryPlan, method="json_schema")
    return _PLANNER_CHAIN


async def plan_query_execution(user_query: str) -> QueryPlan:
    """Uses Gemini structured output to parse intent, extract multiple entities, and prune unnecessary tools."""
    planner = _get_planner_chain()
    prompt = (
        "Analyze this financial market query and extract an execution plan. "
        "Expand common Indian/global stock shorthand into full searchable names "
        "(e.g., 'M&M' -> 'Mahindra & Mahindra', 'RIL' -> 'Reliance Industries', 'Infy' -> 'Infosys').\n\n"
        f"Query: {user_query}"
    )
    return await planner.ainvoke(prompt)


async def resolve_ticker(company_query: str, prefer_india: bool = True) -> Dict[str, Any]:
    """Resolves a company name into an exact NSE/BSE or global ticker symbol."""
    def _resolve() -> Dict[str, Any]:
        import yfinance as yf  # Lazy import so cold startup doesn't freeze

        clean_query = company_query.strip()
        try:
            search = yf.Search(clean_query, max_results=8)
            quotes = [q for q in search.quotes if q.get("quoteType") == "EQUITY"]

            if not quotes:
                return {"error": f"Could not find any stock matching '{company_query}'."}

            selected = None
            if clean_query.upper().endswith((".NS", ".BO")):
                for q in quotes:
                    if q.get("symbol", "").upper() == clean_query.upper():
                        selected = q
                        break

            if not selected and prefer_india:
                nse_matches = [q for q in quotes if q.get("symbol", "").endswith(".NS")]
                bse_matches = [q for q in quotes if q.get("symbol", "").endswith(".BO")]
                if nse_matches:
                    selected = nse_matches[0]
                elif bse_matches:
                    selected = bse_matches[0]

            if not selected:
                selected = quotes[0]

            symbol = selected.get("symbol", "")
            raw_name = selected.get("longname") or selected.get("shortname") or clean_query
            wiki_name = (
                raw_name.replace(" Limited", "")
                .replace(" Ltd.", "")
                .replace(" Ltd", "")
                .replace(", Inc.", "")
                .replace(" Inc.", "")
                .replace(" Corporation", "")
                .strip()
            )
            return {
                "ticker": symbol,
                "company_name": raw_name,
                "wiki_query": wiki_name,
                "exchange": selected.get("exchDisp", "Unknown"),
                "is_indian_stock": symbol.endswith((".NS", ".BO")),
            }
        except Exception as e:
            return {"error": f"Ticker resolution failed for '{company_query}': {str(e)}"}

    return await asyncio.to_thread(_resolve)