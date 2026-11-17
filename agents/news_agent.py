import asyncio
import urllib.parse
from typing import List, Dict
import feedparser


async def fetch_news_sentiment(
    company_name: str,
    is_indian_stock: bool = False,
    max_results: int = 5
) -> List[Dict[str, str]]:
    """Fetches real-time financial news via Google News RSS (zero rate limits)."""
    def _fetch() -> List[Dict[str, str]]:
        try:
            encoded_query = urllib.parse.quote(f"{company_name} stock news")
            if is_indian_stock:
                url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
            else:
                url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

            feed = feedparser.parse(url)
            articles = []
            for entry in feed.entries[:max_results]:
                source = entry.source.title if hasattr(entry, "source") else "News"
                articles.append({
                    "title": entry.get("title", ""),
                    "source": source,
                    "published": entry.get("published", ""),
                })
            return articles if articles else [{"info": f"No recent headlines found for {company_name}."}]
        except Exception as e:
            return [{"error": f"Failed to fetch news for {company_name}: {str(e)}"}]

    return await asyncio.to_thread(_fetch)