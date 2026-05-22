import os
import httpx
from mcp.server.fastmcp import FastMCP

# NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
# NEWSAPI_URL = "https://newsapi.org/v2/everything"

WORLDNEWS_API_KEY = os.getenv("WORLDNEWS_KEY")
WORLDNEWS_URL = "https://api.worldnewsapi.com/search-news"

mcp = FastMCP("mcp-news-server")

@mcp.tool()
async def fetch_news_stream(
    query : str,
    language : str = "en",
    page_size : int = 10
) -> str:
    '''
    Search and fetch real-time news articles from NewsAPI based on the LLM's extracted keywords.
    '''
    if not WORLDNEWS_API_KEY:
        return "Error. Key missing."

    params = {
        "text" : query,
        "language" : language,
        "number" : min(max(page_size, 1), 100),
        "api-key" : WORLDNEWS_API_KEY,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(WORLDNEWS_URL, params=params)

            if response.status_code != 200:
                return f"HTTP API Failure: Endpoint returned error code {response.status_code}."

            data = response.json()
            articles = data.get("news", [])

            if not articles:
                return f"Query returned empty. No articles found matching: '{query}'."

            compiled_report = []
            for index, item in enumerate(articles, 1):
                title = item.get("title", "Untitled Article")
                text = item.get("text", "No content abstract available.")
                summary = item.get("summary", "No summary available.")
                url = item.get("url", "No Link")
                # image = item.get("image", "No image")
                source = ", ".join(item.get("authors", [])) or "Unknown Author"

                news_block = (
                    f"### [Article #{index}] {title}\n"
                    f"**Author(s):** {source} | **URL:** {url}\n"
                    f"**Summary:** {summary}\n"
                    f"**Full Content:** {text}\n"
                    f"{'='*50}"
                )
                compiled_report.append(news_block)

            return "\n\n".join(compiled_report)

    except httpx.RequestError as error:
        return f"Network layer exception encountered while contacting provider: {str(error)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
