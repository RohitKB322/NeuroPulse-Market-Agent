import asyncio
import wikipediaapi


async def fetch_company_fundamentals(company_name: str) -> str:
    """Pulls encyclopedic background context from Wikipedia."""
    def _fetch() -> str:
        try:
            wiki = wikipediaapi.Wikipedia(
                user_agent="FinancialMarketAgent/1.0 (CLI-Prototype)",
                language="en"
            )
            page = wiki.page(company_name)
            if page.exists():
                return page.summary[:700]
            return f"No Wikipedia summary found for '{company_name}'."
        except Exception as e:
            return f"Error fetching Wikipedia context: {str(e)}"

    return await asyncio.to_thread(_fetch)