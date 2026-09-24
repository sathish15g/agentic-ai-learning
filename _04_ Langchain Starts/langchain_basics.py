"""A LangChain agent with real tools, fetching a real book from Project Gutenberg.

Concept: don't ask the model to eyeball-count matches inside a wall of text
you pasted into its context -- that's exactly the kind of thing LLMs get
wrong on long documents. Instead, give it tools that do the counting in
real code and hand back a trustworthy number. The model's job is choosing
which tool to call and turning results into prose, not doing arithmetic
on raw text itself.

pip install -qU langchain "langchain[openai]" python-dotenv
.env: GROQ_API_KEY=...
"""

import os
import urllib.error
import urllib.request

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

SYSTEM_PROMPT = """You are a literary data assistant.

## Capabilities

- `fetch_text_from_url`: loads document text from a URL so you can read and summarize it.
- `count_lines_containing`: exact count of lines in a URL's document containing a substring.
- `first_line_number_containing`: 1-based line number of the first line containing a substring.

Always use the counting tools for exact numbers -- never estimate a count yourself
by reading the fetched text. If a tool reports an error, say so plainly rather than
guessing a number."""


def _fetch(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; quickstart-research/1.0)"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()
    return raw.decode("utf-8", errors="replace")


@tool
def fetch_text_from_url(url: str) -> str:
    """Fetch the full document text from a URL, for reading/summarizing."""
    try:
        return _fetch(url)
    except urllib.error.URLError as e:
        return f"Fetch failed: {e}"


@tool
def count_lines_containing(url: str, substring: str) -> str:
    """Fetch the document at url and count how many lines contain substring
    (counts matching lines, not total occurrences -- a line with the
    substring twice still counts once)."""
    try:
        text = _fetch(url)
    except urllib.error.URLError as e:
        return f"Fetch failed: {e}"
    count = sum(1 for line in text.splitlines() if substring in line)
    return f"{count} lines contain {substring!r}"


@tool
def first_line_number_containing(url: str, substring: str) -> str:
    """Fetch the document at url and return the 1-based line number of the
    first line containing substring, or a not-found message."""
    try:
        text = _fetch(url)
    except urllib.error.URLError as e:
        return f"Fetch failed: {e}"
    for line_number, line in enumerate(text.splitlines(), start=1):
        if substring in line:
            return f"Line {line_number}"
    return f"{substring!r} not found in document"


llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.5,
    timeout=300,
)

agent = create_agent(
    model=llm,
    tools=[fetch_text_from_url, count_lines_containing, first_line_number_containing],
    system_prompt=SYSTEM_PROMPT,
)

content = """Project Gutenberg hosts a full plain-text copy of F. Scott Fitzgerald's The Great Gatsby.
URL: https://www.gutenberg.org/files/64317/64317-0.txt

Answer:
1) How many lines in the file contain the substring `Gatsby`?
2) The 1-based line number of the first line that contains `Daisy`.
3) A two-sentence neutral synopsis.

Use your tools for (1) and (2) -- do not estimate. If a tool call fails, report the
error rather than fabricating a number."""

if __name__ == "__main__":
    agent_result = agent.invoke(
        {"messages": [{"role": "user", "content": content}]},
        config={"configurable": {"thread_id": "great-gatsby-lc"}},
    )
    print(agent_result["messages"][-1].content)
