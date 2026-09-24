"""Letting the model choose a tool for itself, via a tool schema.

File 4 had US deciding which tool to call. Here we hand the model a menu
of tools (schemas only, not the functions themselves) and let IT decide --
by returning a `tool_calls` list instead of plain text when it thinks a
tool applies. We still execute the call ourselves; the model only picks.

Needs an OpenAI-compatible provider (Groq, OpenRouter, or OpenAI) --
Anthropic's tool-calling API uses a different response shape, covered
in a later module.

Setup: uv add openai python-dotenv
.env:  set at least one of GROQ_API_KEY / OPENROUTER_API_KEY / OPENAI_API_KEY
"""

import json
import os

from dotenv import load_dotenv

load_dotenv()


# --- tools: plain functions, none of them aware an AI exists ---

SAMPLE_WEATHER = {
    "tokyo": {"celsius": 22, "conditions": "partly cloudy"},
    "delhi": {"celsius": 34, "conditions": "clear skies"},
    "london": {"celsius": 15, "conditions": "light rain"},
}


def get_weather(city: str) -> str:
    data = SAMPLE_WEATHER.get(city.lower())
    if data is None:
        return f"No weather data for {city!r}."
    return f"{city.title()}: {data['celsius']}C, {data['conditions']}"


CAPITALS = {
    "india": "New Delhi",
    "japan": "Tokyo",
    "france": "Paris",
    "usa": "Washington DC",
    "uk": "London",
}


def get_capital(country: str) -> str:
    return CAPITALS.get(country.lower(), f"No capital on file for {country!r}.")


def calculator(expression: str) -> str:
    """Only digits, operators, and parentheses are allowed through before
    eval() ever runs, so arbitrary code can't be smuggled in via the
    expression string the model hands back.
    """
    allowed_characters = set("0123456789+-*/(). ")
    if not set(expression) <= allowed_characters:
        return f"Rejected -- disallowed characters in {expression!r}."
    try:
        return str(eval(expression))  # noqa: S307 -- input whitelisted above
    except Exception as exc:  # noqa: BLE001
        return f"Could not evaluate: {exc}"


# name -> function, so a chosen tool_call can be dispatched without an
# if/elif chain. Every entry here needs a matching schema below.
TOOLS_BY_NAME = {
    "get_weather": get_weather,
    "get_capital": get_capital,
    "calculator": calculator,
}


# --- the "menu" handed to the model. It never sees the functions above --
# only these descriptions. The wording of "description" is what tells the
# model when each tool is relevant to a given question. ---

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city. Use this whenever "
                            "the user asks about weather, temperature, or conditions "
                            "in a specific place.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city name, e.g. 'Tokyo'."}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_capital",
            "description": "Get the capital city of a country. Use this whenever "
                            "the user asks about the capital of a specific country.",
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {"type": "string", "description": "The country name, e.g. 'France'."}
                },
                "required": ["country"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform a calculation. Use this whenever the user asks "
                            "for a calculation or a math problem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "e.g. '(1 + 2) * 45'."}
                },
                "required": ["expression"],
            },
        },
    },
]


def get_client_and_model():
    """Picks whichever OpenAI-compatible provider has a key set. Raises
    clearly if none is configured, since real decision-making genuinely
    needs a real model to call.
    """
    from openai import OpenAI

    if os.environ.get("GROQ_API_KEY"):
        return (
            OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1"),
            "openai/gpt-oss-120b",
        )
    if os.environ.get("OPENROUTER_API_KEY"):
        return (
            OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1"),
            "openrouter/free",
        )
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"]), "gpt-4o-mini"

    raise RuntimeError(
        "No OpenAI-compatible key found. Set one of GROQ_API_KEY, "
        "OPENROUTER_API_KEY, or OPENAI_API_KEY in your .env file."
    )


def ask_ai_to_choose(question: str):
    """Sends the question plus every tool schema in one call. The reply may
    contain a tool_calls list instead of plain text -- that list is the
    model's DECISION, not an executed result. Nothing has run yet.
    """
    client, model = get_client_and_model()
    response = client.chat.completions.create(
        model=model,
        max_tokens=300,
        messages=[{"role": "user", "content": question}],
        tools=TOOL_SCHEMAS,
    )
    return response.choices[0].message


if __name__ == "__main__":
    # Four questions, four different outcomes -- this is the actual point
    # of the file: the SAME code path handles all of them, because the
    # decision of which tool (if any) applies is made by the model, not
    # hardcoded per question like it was in File 4.
    questions = [
        "What is the weather in Tokyo right now?",
        "What is the capital of Japan?",
        "What is (1 + 2) * 45?",
        "In one sentence, why do software teams write tests?",  # no tool fits
    ]

    for question in questions:
        print(f"\nQ: {question}")
        message = ask_ai_to_choose(question)

        if message.tool_calls:
            for call in message.tool_calls:
                arguments = json.loads(call.function.arguments)
                tool_function = TOOLS_BY_NAME[call.function.name]
                result = tool_function(**arguments)
                print(f"  -> chose tool {call.function.name}({arguments}) = {result}")
        else:
            print(f"  -> no tool needed: {message.content}")
