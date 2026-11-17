import asyncio
from typing import Dict, Any
import yfinance as yf


async def fetch_quantitative_metrics(ticker: str) -> Dict[str, Any]:
    """Fetches key stock metrics asynchronous"""
    def _fetch() -> Dict[str, Any]:
        try:
            stock = yf.Ticker(ticker)
            info = stock.fast_info
            return {
                "last_price": round(info.last_price, 2) if getattr(info, "last_price", None) else None,
                "currency": getattr(info, "currency", "USD"),
                "market_cap": getattr(info, "market_cap", None),
                "fifty_two_week_high": round(info.year_high, 2) if getattr(info, "year_high", None) else None,
                "fifty_two_week_low": round(info.year_low, 2) if getattr(info, "year_low", None) else None,
            }
        except Exception as e:
            return {"error": f"Failed to fetch metrics for {ticker}: {str(e)}"}

    return await asyncio.to_thread(_fetch)