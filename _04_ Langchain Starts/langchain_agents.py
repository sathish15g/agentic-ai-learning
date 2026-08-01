# pip install -qU langchain "langchain[openai]"
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
import pprint
load_dotenv()
import os

import urllib.error
import urllib.request

from langchain.tools import tool


@tool
def fetch_text_from_url(url: str) -> str:
    """Fetch the document from a URL.
    """
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; quickstart-research/1.0)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
    except urllib.error.URLError as e:
        return f"Fetch failed: {e}"
    text = raw.decode("utf-8", errors="replace")
    return text

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


llm = ChatOpenAI(
    model="openrouter/free",          # or another OpenRouter model
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

agent = create_agent(
    model=llm,
    tools=[get_weather, fetch_text_from_url],
    system_prompt="You are a helpful assistant",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is weather in Delhi?"}]}
)

pprint.pprint(result)
print(result["messages"][-1].content_blocks)