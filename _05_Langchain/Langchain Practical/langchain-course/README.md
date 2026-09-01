# LangChain Course

A hands-on LangChain workspace, managed with [uv](https://docs.astral.sh/uv/).

## Setup

```bash
uv sync
```

Create a `.env` file (see `.gitignore` — it's never committed) with the API keys the
notebooks expect, e.g. `OPENAI_API_KEY`.

## How `Learning/` is organized

Each `Part N - <Topic>` folder is one step in the course, in order. Inside a part folder:

- `*.ipynb` without a suffix — hands-on practice notebooks written while working through that part.
- `reference-notes.ipynb` — reference/theory notes for that part (imported from external course notes), kept alongside the practice notebook it corresponds to.

| Folder | Covers |
|---|---|
| `Part 0 - Overview` | The LangChain/LangGraph/LangSmith ecosystem, why the layers exist, FAQs |
| `Part 1 - Environment Setup` | Installing dependencies, verifying the environment works |
| `Part 2 - Models` | Calling chat models, streaming, tool binding, structured model output, messages |
| `Part 3 - Prompt Templates and Structured Output` | `ChatPromptTemplate`, `MessagesPlaceholder`, structured output patterns, with a preview of Tools & Agents (covered in later parts, not yet added here) |
| `Revision` | Cumulative revision notes spanning Parts 1-4 |

Note: `Part 2`'s reference notes also recap environment setup and messages, since the
source material grouped those topics together.
