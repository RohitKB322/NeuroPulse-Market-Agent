import asyncio
from typing import Dict, Any
import yfinance as yf


async def resolve_ticker(user_query: str, prefer_india: bool = True) -> Dict[str, Any]:
    """
    Resolves a natural company name or partial ticker (e.g., 'Tata Motors', 'Reliance', 'NVDA')
    into an exact ticker symbol, clean company name, and exchange metadata.
    """
    def _resolve() -> Dict[str, Any]:
        clean_query = user_query.strip()
        try:
            search = yf.Search(clean_query, max_results=8)
            quotes = [q for q in search.quotes if q.get("quoteType") == "EQUITY"]

            if not quotes:
                return {"error": f"Could not find any stock matching '{user_query}'."}

            selected = None

            # 1. If user explicitly typed .NS or .BO, match exact symbol
            if clean_query.upper().endswith((".NS", ".BO")):
                for q in quotes:
                    if q.get("symbol", "").upper() == clean_query.upper():
                        selected = q
                        break

            # 2. Prefer Indian exchanges (NSE first, then BSE) if prefer_india is True
            if not selected and prefer_india:
                nse_matches = [q for q in quotes if q.get("symbol", "").endswith(".NS")]
                bse_matches = [q for q in quotes if q.get("symbol", "").endswith(".BO")]
                if nse_matches:
                    selected = nse_matches[0]
                elif bse_matches:
                    selected = bse_matches[0]

            # 3. Fallback to top global equity match (e.g., NVDA, AAPL)
            if not selected:
                selected = quotes[0]

            symbol = selected.get("symbol", "")
            raw_name = selected.get("longname") or selected.get("shortname") or clean_query
            # Clean corporate suffixes so Wikipedia finds the exact page easily
            wiki_name = (
                raw_name.replace(" Limited", "")
                .replace(" Ltd.", "")
                .replace(" Ltd", "")
                .replace(", Inc.", "")
                .replace(" Inc.", "")
                .replace(" Corporation", "")
                .strip()
            )
            is_indian = symbol.endswith((".NS", ".BO"))

            return {
                "ticker": symbol,
                "company_name": raw_name,
                "wiki_query": wiki_name,
                "exchange": selected.get("exchDisp", "Unknown"),
                "is_indian_stock": is_indian,
            }
        except Exception as e:
            return {"error": f"Ticker resolution failed: {str(e)}"}

    return await asyncio.to_thread(_resolve)