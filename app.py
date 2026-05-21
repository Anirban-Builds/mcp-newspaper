import os
from fastapi import FastAPI,HTTPException
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import uvicorn
from src.ask_llm import LLMNewsEngine
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

mcp_session = None
ai_engine = None

@asynccontextmanager
async def _run_mcp_server(app: FastAPI):
    global ai_engine, mcp_session
    ai_engine = LLMNewsEngine()

    server_params = StdioServerParameters(
        command="python",
        args= ["src/mcp_server.py"],
        env= os.environ.copy()
    )
    transport = stdio_client(server_params)
    read_stream, write_stream = await transport.__aenter__()
    try :
        session = ClientSession(read_stream, write_stream)
        await session.__aenter__()
        try :
            await session.initialize()
            mcp_session = session
            print("MCP server running\n")
            yield
        finally :
            await session.__aexit__(None, None, None)
    finally:
        await transport.__aexit__(None, None, None)

app = FastAPI(title="mcp-news-app", lifespan=_run_mcp_server)

@app.get("/")
async def root():
    return {"status": "ok", "message": "Space is running"}

@app.get("/ask-news")
async def handle_ask_news(user_prompt: str):
    global mcp_session, ai_engine

    if not mcp_session or not ai_engine:
        raise HTTPException(
            status_code=503,
            detail="AI pipelines are still starting up."
        )

    try:
        reply = await ai_engine.llm_response(user_prompt, mcp_session)
        return {"success": True, "response": reply}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline Processing Fault: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

