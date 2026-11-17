import os
import json
from typing import Dict, Any, AsyncGenerator
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

SYSTEM_INSTRUCTION = """You are an institutional-grade Equity Research & Market Sentiment Analyst.
You will receive a JSON state object containing live quantitative metrics, recent news headlines, and a corporate overview for a public company.

Synthesize a crisp, high-signal market intelligence brief using this exact structure:
1. **Snapshot & Price Action**: Current price, 52-week range position (calculate % from 52W high/low), and formatted Market Cap (use ₹ Lakh Crore / Crore for INR, and $B / $T for USD).
2. **News & Catalyst Sentiment**: Classify overall short-term sentiment (Bullish / Bearish / Neutral / Mixed) with a 1-line rationale, followed by 2–3 bullet points highlighting the key drivers from the headlines.
3. **Business Fundamentals**: 2 concise sentences summarizing core revenue drivers and market positioning.
4. **Analyst Takeaway**: One sharp, objective takeaway on valuation/momentum context (no generic financial disclaimers).

Keep the response scannable, data-dense, and under 250 words."""


async def stream_financial_synthesis(
    state: Dict[str, Any],
    model_id: str = "gemini-3.5-flash-lite"
) -> AsyncGenerator[str, None]:
    """
    Streams the synthesized financial brief token-by-token using the async GenAI client.
    Designed as an AsyncGenerator so it works identically in CLI and future FastAPI SSE endpoints.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        yield "\n[Error] GOOGLE_API_KEY not found in .env file."
        return

    client = genai.Client(api_key=api_key)

    prompt = (
        f"Analyze the following real-time market intelligence state for "
        f"{state.get('company_name')} ({state.get('ticker')}):\n\n"
        f"{json.dumps(state, indent=2)}"
    )

    try:
        response_stream = await client.aio.models.generate_content_stream(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.3,
            ),
        )

        async for chunk in response_stream:
            if chunk.text:
                yield chunk.text

    except Exception as e:
        yield f"\n[Synthesis Error] Failed to stream response from {model_id}: {str(e)}"