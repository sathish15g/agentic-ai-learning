import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# -------------------------------------------------------------------
# Fake weather database
# -------------------------------------------------------------------

SAMPLE_WEATHER = {
    "tokyo": {"celsius": 22, "conditions": "Partly Cloudy"},
    "delhi": {"celsius": 34, "conditions": "Clear Skies"},
    "london": {"celsius": 15, "conditions": "Light Rain"},
}


# -------------------------------------------------------------------
# TOOLS
# -------------------------------------------------------------------

def get_weather(city: str) -> str:
    """Return weather for a city."""

    weather = SAMPLE_WEATHER.get(city.lower())

    if weather is None:
        return f"No weather data found for {city}"

    return (
        f"Weather in {city.title()}:\n"
        f"Temperature : {weather['celsius']}°C\n"
        f"Conditions  : {weather['conditions']}"
    )


def get_capital(country: str) -> str:
    capitals = {
        "india": "New Delhi",
        "japan": "Tokyo",
        "france": "Paris",
        "usa": "Washington DC",
        "uk": "London",
    }

    return capitals.get(country.lower(), "Capital not found")


def calculator(expression: str) -> str:
    """Only digits, operators, and parentheses are allowed through before
    eval() ever runs, so arbitrary code can't be smuggled in via a string
    the model hands back.
    """
    allowed_characters = set("0123456789+-*/(). ")
    if not set(expression) <= allowed_characters:
        return f"Rejected -- disallowed characters in {expression!r}."
    try:
        return str(eval(expression))  # noqa: S307 -- input whitelisted above
    except Exception as e:
        return str(e)


TOOLS_BY_NAME = {
    "get_weather": get_weather,
    "get_capital": get_capital,
    "calculator": calculator,
}


# -------------------------------------------------------------------
# Tool Schemas
# -------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather of a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_capital",
            "description": "Get capital city of a country.",
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {
                        "type": "string",
                        "description": "Country name",
                    }
                },
                "required": ["country"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform mathematical calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]


# -------------------------------------------------------------------
# Client
# -------------------------------------------------------------------

def get_client_and_model():

    if os.environ.get("GROQ_API_KEY"):

        print("Using Groq")

        return (
            OpenAI(
                api_key=os.environ["GROQ_API_KEY"],
                base_url="https://api.groq.com/openai/v1",
            ),
            "openai/gpt-oss-120b",
        )

    if os.environ.get("OPENROUTER_API_KEY"):

        print("Using OpenRouter")

        return (
            OpenAI(
                api_key=os.environ["OPENROUTER_API_KEY"],
                base_url="https://openrouter.ai/api/v1",
            ),
            "openrouter/free",
        )

    if os.environ.get("OPENAI_API_KEY"):

        print("Using OpenAI")

        return (
            OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
            "gpt-4o-mini",
        )

    raise RuntimeError("No API Key Found")


# -------------------------------------------------------------------
# Execute tool calls
# -------------------------------------------------------------------

def execute_tool_calls(tool_calls, messages):

    for tool_call in tool_calls:

        function_name = tool_call.function.name

        arguments = json.loads(tool_call.function.arguments)

        function = TOOLS_BY_NAME[function_name]

        result = function(**arguments)

        print(f"\nTool Called : {function_name}")
        print(f"\nTool id : {tool_call.id}")
        print(f"Arguments   : {arguments}")
        print(f"Result      : {result}\n")

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
        )


# -------------------------------------------------------------------
# Agent
# -------------------------------------------------------------------

def run_agent(messages, max_turns=5):

    client, model = get_client_and_model()

    for _ in range(max_turns):

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
        )

        assistant = response.choices[0].message

        if assistant.tool_calls is None:

            messages.append(
                {
                    "role": "assistant",
                    "content": assistant.content,
                }
            )

            return assistant.content

        messages.append(
            {
                "role": "assistant",
                "content": assistant.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in assistant.tool_calls
                ],
            }
        )

        execute_tool_calls(assistant.tool_calls, messages)

    return "Reached maximum iterations."


# -------------------------------------------------------------------
# Chat
# -------------------------------------------------------------------

def chat():

    print("=" * 60)
    print("Project Zero Agent")
    print("Type 'exit' to quit")
    print("=" * 60)

    conversation = []

    while True:

        user = input("\nYou : ")

        if user.lower() in ["exit", "quit"]:

            break

        conversation.append(
            {
                "role": "user",
                "content": user,
            }
        )

        answer = run_agent(conversation)

        print("\nAgent:", answer)


if __name__ == "__main__":
    chat()

"""
$ python _06_project_zero_agent.py
============================================================
Project Zero Agent
Type 'exit' to quit
============================================================

You : what is weather of tokyo?
Using Groq

Tool Called : get_weather

Tool id : fc_b59dc92f-d2d1-4d66-9066-4d2ff3e572c6
Arguments   : {'city': 'Tokyo'}
Result      : Weather in Tokyo:
Temperature : 22°C
Conditions  : Partly Cloudy


Agent: Here's the latest weather for Tokyo: 22°C, partly cloudy.

You : what is capital of India?
Using Groq

Tool Called : get_capital

Tool id : fc_cb3ba726-d896-4439-8643-a2e8e6d5feac
Arguments   : {'country': 'India'}
Result      : New Delhi


Agent: The capital of India is **New Delhi**.

You : calculate (1+2)*45
Using Groq

Tool Called : calculator

Tool id : fc_e264897d-7e41-4ad6-b1c2-42e4e52b97b1
Arguments   : {'expression': '(1+2)*45'}
Result      : 135


Agent: The result of (1+2) * 45 is **135**.

You : exit
"""