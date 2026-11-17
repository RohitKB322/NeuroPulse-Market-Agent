import asyncio
from typing import Dict, Any, List, Literal, Optional
from pydantic import BaseModel, Field


class QueryPlan(BaseModel):
    intent: Literal["single_analysis", "comparison", "news_only", "follow_up"] = Field(
        description=(
            "'follow_up' if the user is asking a follow-up question about companies ALREADY present in cached_companies; "
            "'comparison' if comparing 2+ companies/stocks; "
            "'news_only' if asking only about recent news or why a stock moved today; "
            "'single_analysis' for analyzing a new single stock."
        )
    )
    companies_to_fetch: List[str] = Field(
        description=(
            "List of NEW corporate names or tickers that need fresh data fetched. "
            "Leave empty [] if intent is 'follow_up' and all needed companies are already in cached_companies."
        )
    )
    need_fundamentals: bool = Field(
        description="False if intent is 'news_only' or 'follow_up'; True for new full stock analyses."
    )
    focus_aspect: Optional[str] = Field(
        default=None,
        description="Specific angle requested by user (e.g., 'valuation', 'momentum', 'risks'), or None."
    )


_PLANNER_CHAIN = None


def _get_planner_chain():
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


async def plan_query_execution(user_query: str, cached_companies: List[str] = None) -> QueryPlan:
    """Parses intent, checks cached memory, and determines which new workers (if any) to spawn."""
    planner = _get_planner_chain()
    cached_str = ", ".join(cached_companies) if cached_companies else "None"
    prompt = (
        "Analyze this financial chat message and decide the execution plan.\n"
        f"Companies already cached in this chat thread: [{cached_str}]\n\n"
        "Rules:\n"
        "- If the user asks a follow-up about companies already cached (e.g., 'which one has better momentum?', 'summarize risks'), "
        "set intent='follow_up' and companies_to_fetch=[].\n"
        "- If the user asks to add a new company to an existing comparison (e.g., 'now add Maruti'), "
        "set intent='comparison' and put ONLY the new company in companies_to_fetch.\n"
        "- Expand shorthand (e.g., 'M&M' -> 'Mahindra & Mahindra', 'RIL' -> 'Reliance Industries').\n\n"
        f"User Query: {user_query}"
    )
    return await planner.ainvoke(prompt)


async def resolve_ticker(company_query: str, prefer_india: bool = True) -> Dict[str, Any]:
    """Resolves a company name into an exact NSE/BSE or global ticker symbol."""
    def _resolve() -> Dict[str, Any]:
        import yfinance as yf

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