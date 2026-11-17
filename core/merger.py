import time
from typing import Dict, Any, List


def merge_agent_outputs(
    ticker: str,
    company: str,
    quant: Dict[str, Any],
    news: List[Dict[str, str]],
    summary: str
) -> Dict[str, Any]:
    """Merges isolated sub-agent outputs into a unified global state."""
    return {
        "ticker": ticker.upper(),
        "company_name": company,
        "metrics": quant,
        "news_headlines": news,
        "business_overview": summary,
        "timestamp": round(time.time(), 2)
    }