import asyncio
from typing import Dict, Any


async def fetch_quantitative_metrics(ticker: str) -> Dict[str, Any]:
    """Fetches price, daily % change, 52W range, and valuation metrics asynchronously."""
    def _fetch() -> Dict[str, Any]:
        import yfinance as yf

        try:
            stock = yf.Ticker(ticker)
            fast = stock.fast_info

            last_price = getattr(fast, "last_price", None)
            prev_close = getattr(fast, "previous_close", None)
            day_change_pct = None
            if last_price and prev_close:
                day_change_pct = round(((last_price - prev_close) / prev_close) * 100, 2)

            # Lightweight check for P/E and sector without heavy scraping
            info = stock.info or {}
            trailing_pe = info.get("trailingPE")

            return {
                "last_price": round(last_price, 2) if last_price else None,
                "day_change_pct": day_change_pct,
                "currency": getattr(fast, "currency", "USD"),
                "market_cap": getattr(fast, "market_cap", None),
                "trailing_pe": round(trailing_pe, 2) if isinstance(trailing_pe, (int, float)) else None,
                "sector": info.get("sector", "N/A"),
                "fifty_two_week_high": round(fast.year_high, 2) if getattr(fast, "year_high", None) else None,
                "fifty_two_week_low": round(fast.year_low, 2) if getattr(fast, "year_low", None) else None,
            }
        except Exception as e:
            return {"error": f"Failed to fetch metrics for {ticker}: {str(e)}"}

    return await asyncio.to_thread(_fetch)