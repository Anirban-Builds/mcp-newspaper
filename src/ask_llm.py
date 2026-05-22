import os
from google import genai
from google.genai import types
from datetime import datetime

class LLMNewsEngine:
    def __init__(self,
                 model_name= "gemma-4-26b-a4b-it"):
        self.client = genai.Client(api_key=os.getenv("LLM_API_KEY"))
        self.model_name = model_name

    def _get_news_tool_schema(self) ->list:
        return [{"function_declarations":[
            {
                "name": "fetch_news_stream",
                "description": "Searches real-time global news articles using complex keyword strings and boolean logic.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "query": {
                            "type": "STRING",
                            "description": "Clean search query strings. Supports boolean rules, e.g., 'EV AND Germany NOT stock'"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]}]
    async def llm_response(self, prompt : str, mcp_session)->str:

        current_date_str = datetime.now().strftime("%A, %B %d, %Y")

        system_instruction = (
            f"You are a real-time news assistant. The current real-world date is {current_date_str}. "
            f"Do not treat the current year or date as the future. It is the present. "
            f"Always use the provided fetch_news_stream tool to pull up-to-date information."
        )

        init_res = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=self._get_news_tool_schema(),
                system_instruction=system_instruction),
        )

        if init_res.function_calls:
            tool_call = init_res.function_calls[0]
            ext_query = tool_call.args["query"]

            mcp_res = await mcp_session.call_tool(
                "fetch_news_stream",
                arguments={"query": ext_query, "page_size" : 10}
            )
            raw_res = mcp_res.content[0].text
            final_res = self.client.models.generate_content(
                model=self.model_name,
                contents=(
                    f"You are an expert news curator and investigative journalist. Turn this raw news feed into a comprehensive, "
                    f"highly detailed briefing matching the user's initial request.\n\n"
                    f"USER ORIGINAL INTENT: {prompt}\n\n"
                    f"RAW DATA:\n{raw_res}\n\n"
                    f"CRITICAL INSTRUCTIONS FOR RETAINING MAXIMUM TEXT AND DETAIL:\n"
                    f"- Do NOT overly condense, abstract, or sanitize the raw data. Your goal is maximum information retention.\n"
                    f"- Maintain specific facts, exact numbers, dates, quotes, names, and chronological details from the raw feed.\n"
                    f"- If the raw data contains multiple distinct news stories or updates, cover EACH one thoroughly in its own section. Do not merge separate events into generic summaries.\n"
                    f"- Aim for an exhaustive, deep-dive synthesis rather than a brief overview.\n\n"
                    f"IMPORTANT: At the end of your response, include a '## Sources' section "
                    f"listing all article titles with their URLs in markdown link format: "
                    f"[Article Title](URL). Only include links that are present in the raw data."
                    f"IMPORTANT: At the end of your response, include a '## Sources' section "
                    f"listing all article titles with their URLs in markdown link format: "
                    f"[Article Title](URL). Only include links that are explicitly present in the raw data. "
                    f"Do not invent or guess any URLs."
                )
            )
            return final_res.text
        return init_res.text