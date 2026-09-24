"""A LangChain agent choosing between two real tools: live weather and URL fetch.

Concept: create_agent() wires up the choose -> call -> observe -> repeat loop
for you (the same loop File 6 of Project 0 built by hand). Handing it two
very different tools -- one hitting a live weather API, one fetching a raw
document -- shows the model picking the right one per question, not just
always calling the one tool available.

pip install -qU langchain "langchain[openai]" python-dotenv requests
.env: GROQ_API_KEY=...
"""

import os
import urllib.error
import urllib.request

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()


@tool
def fetch_text_from_url(url: str) -> str:
    """Fetch the raw text of a document from a URL."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; quickstart-research/1.0)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
    except urllib.error.URLError as e:
        return f"Fetch failed: {e}"
    return raw.decode("utf-8", errors="replace")


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city, using live data (Open-Meteo, no API key)."""
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=10,
        ).json()
        if not geo.get("results"):
            return f"Could not find a location named {city!r}."
        place = geo["results"][0]
        forecast = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,weather_code",
            },
            timeout=10,
        ).json()["current"]
        return f"{place['name']}, {place['country']}: {forecast['temperature_2m']}C"
    except requests.exceptions.RequestException as exc:
        return f"Weather service unavailable: {exc}"


llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)

agent = create_agent(
    model=llm,
    tools=[get_weather, fetch_text_from_url],
    system_prompt="You are a helpful assistant.",
)

if __name__ == "__main__":
    for question in [
        "What is the weather in Delhi?",
        "Fetch https://raw.githubusercontent.com/octocat/Hello-World/master/README and tell me exactly what it says.",
    ]:
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        print(f"Q: {question}\nA: {result['messages'][-1].content}\n")
