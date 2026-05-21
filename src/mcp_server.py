import os
import httpx
from mcp.server.fastmcp import FastMCP

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
NEWSAPI_URL = "https://newsapi.org/v2/everything"

mcp = FastMCP("mcp-news-server")

@mcp.tool()
async def fetch_news_stream(
    query : str,
    language : str = "en",
    page_size : int = 5
) -> str:
    '''
    Search and fetch real-time news articles from NewsAPI based on the LLM's extracted keywords.
    '''
    if not NEWSAPI_KEY:
        return "Error. Key missing."

    params = {
        "q" : query,
        "language" : language,
        "pageSize" : min(max(page_size, 1), 10),
        "apiKey" : NEWSAPI_KEY,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(NEWSAPI_URL, params=params)

            if response.status_code != 200:
                return f"HTTP API Failure: Endpoint returned error code {response.status_code}."

            data = response.json()
            articles = data.get("articles", [])

            if not articles:
                return f"Query returned empty. No articles found matching: '{query}'."

            compiled_report = []
            for index, item in enumerate(articles, 1):
                title = item.get("title", "Untitled Article")
                source = item.get("source", {}).get("name", "Unknown Publisher")
                description = item.get("description", "No content abstract available.")
                url = item.get("url", "No Link")

                news_block = (
                    f"### [Article #{index}] {title}\n"
                    f"**Publisher:** {source} | **URL:** {url}\n"
                    f"**Abstract:** {description}\n"
                    f"{'='*50}"
                )
                compiled_report.append(news_block)

            return "\n\n".join(compiled_report)

    except httpx.RequestError as error:
        return f"Network layer exception encountered while contacting provider: {str(error)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
