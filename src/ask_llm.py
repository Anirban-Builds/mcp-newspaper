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
            f"You are an expert news curator and a real-time news assistant. "
    f"The current real-world date is {current_date_str}.\n\n"
    f"Always use the provided fetch_news_stream tool to pull up-to-date information.\n\n"
    f"CURATION & FORMATTING INSTRUCTIONS:\n"
    f"- Turn raw news feeds into a cohesive, highly readable summary matching the user's intent.\n"
    f"- CRITICAL: Preserve all layout formats, structures, Markdown tables, and text spacings cleanly. "
    f"Never strip out markdown table syntax pipes (|) or smash structural data into a single dense paragraph.\n"
    f"- You MUST append a 'Quick Reference Sources' section at the very end.\n"
    f"- For every source article referenced from the raw tool response, extract its actual, real-world HTTP/HTTPS URL path.\n"
    f"- Format every reference link strictly using the real extracted external URL directly inside the markdown layout:\n"
    f"  [**[Publisher]** Article Title](ACTUAL_EXTRACTED_URL_HERE)\n"
    f"  Example: [**[BBC News]** Election Updates](https://www.bbc.com/news/articles/123)\n"
    f"- STICK TO REAL LINKS: DO NOT use placeholders, DO NOT use 'source://Source-X', and DO NOT fallback to local domain references. Use the exact external web address provided by the tool."
)
        config=types.GenerateContentConfig(
                tools=self._get_news_tool_schema(),
                system_instruction=system_instruction,
                temperature= 0.3,
        )

        user_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)]
        )

        init_res = self.client.models.generate_content(
            model=self.model_name,
            contents=[user_content],
            config= config,
        )

        if init_res.function_calls:
            tool_call = init_res.function_calls[0]
            ext_query = tool_call.args["query"]

            mcp_res = await mcp_session.call_tool(
                "fetch_news_stream",
                arguments={"query": ext_query}
            )
            raw_res = mcp_res.content[0].text

            model_function_call_content = init_res.candidates[0].content

            tool_response_part = types.Part.from_function_response(
                name=tool_call.name,
                response={"result": raw_res}
            )
            tool_response_content = types.Content(
                role="tool",
                parts=[tool_response_part]
            )

            final_res = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    user_content,
                    model_function_call_content,
                    tool_response_content
                    ],
                config= config
            )
            return final_res.text
        return init_res.text
