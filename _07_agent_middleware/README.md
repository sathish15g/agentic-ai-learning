# 07 · Agent Middleware (LangChain)

Hands-on notes for **LangChain agent middleware**, built around *CineBot*, a movie-ticket assistant.
Middleware lets you control what happens **inside** an agent's loop (limits, approvals, retries, PII
guardrails, context compression) without rewriting the agent, its tools or its prompt.

Everything runs on **Groq's free tier**, so no paid API key is needed.

## What's in this folder

```
_07_agent_middleware/
├── README.md                                      <- you are here
├── Middleware.ipynb                               <- start here - the prebuilt middleware catalog
├── Runtime_Context_and_Custom_Middleware.ipynb     <- companion - Runtime/Context, custom hooks, conditional HITL
├── docs/
│   └── Agent-Middleware-Architecture.pdf          <- the whiteboard: hook lifecycle, HITL, limits, PII
├── archive/                                       <- the two original notebooks Middleware.ipynb was merged from
│   ├── Middleware_v1_colab_openai.ipynb           <-   146 cells, Colab + OpenAI, with saved outputs
│   └── Middleware_v2_wip.ipynb                    <-   159 cells, Groq + SQLite bench, mostly unexecuted
└── cinema.db                                      <- created by the notebooks (SQLite, git-ignored)
```

`Middleware.ipynb` tours the *prebuilt* middleware (summarization, HITL, retries, fallback, PII, ...).
`Runtime_Context_and_Custom_Middleware.ipynb` goes one layer underneath: the `Runtime`/`Context`/`ToolRuntime`
objects those middleware are built on, the two decorator styles for writing your *own* middleware
(node-style `before_model`/`after_model` vs. wrap-style `wrap_model_call`), `dynamic_prompt`, a
conditional (`when`-gated) form of HITL, and a multi-tenant "CineBot Concierge" capstone that combines all
of it. Read `Middleware.ipynb` first; the second notebook assumes its CineBot/`show()` conventions.

## Running it

1. From the repo root: `uv sync`, and make sure `.env` contains `GROQ_API_KEY=...`.
2. Open `Middleware.ipynb` and pick the repo's `.venv` kernel.
3. **Run All.** It takes roughly 6–10 minutes; a few cells deliberately sleep or back off (retry
   demos, and a 60 s pause so Groq's per-minute token budget refills).

To re-run headlessly and refresh the saved outputs (the notebook must run with this folder as its working
directory, because `cinema.db` is created next to it):

```python
import nbformat
from nbclient import NotebookClient

nb = nbformat.read("Middleware.ipynb", as_version=4)
NotebookClient(nb, timeout=900, resources={"metadata": {"path": "."}}).execute()
nbformat.write(nb, "Middleware.ipynb")
```

**If a cell fails with a `429`:** it is Groq's free-tier limits. Wait a minute and re-run that cell.
If it says *tokens per day*, that model's daily budget is spent -- switch `GROQ_MODEL` in section 1.

## The mental model

An agent is a loop: **model -> tools -> model -> ...**. Middleware are hooks around each step:

```
Request
  v
before_agent            once, at the start of the run
  v
+-> before_model        before every model call
|     v
|   wrap_model_call     wraps the model call itself   (retry, fallback, edit the request)
|     v
|   MODEL
|     v
|   after_model         inspect / gate the model's output (e.g. pause before a tool call)
|     v
|   wrap_tool_call      wraps each tool execution     (retry, error handling, emulation)
|     v
|   TOOL ---------------+
+------ loop until the model stops calling tools
  v
after_agent             once, at the end of the run
  v
Response
```

When you stack middleware: `before_*` run first-to-last, `after_*` run last-to-first, and `wrap_*` nest
(the first in the list is the outermost). Middleware is either **prebuilt** (everything below) or
**custom** (section 10 of the notebook writes one with `@wrap_model_call`).

## Middleware covered

The *Hooks* column was read from the installed package (`langchain 1.3.14`) rather than from docs.

| Section | Middleware | Hooks | Use it to... | Key parameters |
|--:|---|---|---|---|
| 3 | `SummarizationMiddleware` | `before_model` | Compress long history into a summary | `trigger`, `keep`, `model`, `summary_prompt` |
| 4 | `HumanInTheLoopMiddleware` | `after_model` | Pause for human approve / edit / reject / respond | `interrupt_on`, needs a checkpointer |
| 5 | `ModelCallLimitMiddleware` | `before_model`, `after_model` | Cap model calls per run / per thread | `run_limit`, `thread_limit`, `exit_behavior` |
| 6 | `ModelFallbackMiddleware` | `wrap_model_call` | Fall back to other models on failure | ordered list of models |
| 7 | `ToolCallLimitMiddleware` | `after_model` | Cap tool calls (all tools, or one by name) | `tool_name`, `run_limit`, `thread_limit` |
| 8 | `PIIMiddleware` | `before_model`, `after_model` | Redact / mask / hash / block personal data | `strategy`, `detector`, `apply_to_*` |
| 9 | `TodoListMiddleware` | `wrap_model_call`, `after_model` | Give the agent a `write_todos` planner | - |
| 10 | `LLMToolSelectorMiddleware` | `wrap_model_call` | Show the model only the relevant tools | `max_tools`, `always_include` |
| 11 | `ToolErrorMiddleware` | `wrap_tool_call` | Turn tool exceptions into recoverable messages | `on_error` |
| 12 | `ToolRetryMiddleware` | `wrap_tool_call` | Retry flaky tools with exponential backoff | `max_retries`, `backoff_factor`, `on_failure` |
| 13 | `LLMToolEmulator` | `wrap_tool_call` | Fake tool results with an LLM, for testing | `tools`, `model` |
| 14 | *(SQLite test bench)* | - | Prove HITL and Summarization against real writes / real context | - |

`ModelRetryMiddleware` appears once (section 9); `ContextEditingMiddleware` / `ClearToolUsesEdit` are
imported but not demonstrated yet (see *Next steps*).

## Things this notebook turned up

Most of these came from running the code rather than reading about it.

- **The HITL `edit` payload is `{"type": "edit", "edited_action": {"name": ..., "args": {...}}}`.** The
  older notebooks used `{"type": "edit", "args": ...}`, which is not the shape in `langchain 1.3.14`.
- **A summariser failure is silent.** If the summary call errors, `SummarizationMiddleware` does not raise:
  it injects the text `Error generating summary: <error>` *as the summary* and drops the old messages.
  This happened for real on Groq (a 429) and the run still "passed". Assert on that string in tests.
- **Summaries are lossy by default.** The default prompt *describes* data ("40 bookings, all confirmed
  except one") instead of *keeping* it. With the default prompt the agent had to call a tool again to
  recover one booking's amount; with a custom `summary_prompt` asking for one compact line per booking it
  answered straight from the summary. Section 14b shows both.
- **A recall test must not leave the answer in the kept messages.** With `keep=("messages", 2)` the
  assistant's own turn-1 table of all 40 bookings survived verbatim, so the "recall" test proved nothing.
  `keep=("messages", 1)` fixes it. (The original test also had no checkpointer, so turn 2 never saw turn 1.)
- **PII masking also hides values from your tools.** `PIIMiddleware("booking_code", strategy="mask")` turned
  `BK1044` into `****1044` *before the model saw it*, so the model called `check_order_status("****1044")`.
  Mask what the model must not see; do not mask identifiers a tool needs to look up.
- **The credit-card detector checks the Luhn checksum.** `4111-1111-1111-1234` (made up) is silently not
  detected; `4111-1111-1111-1111` (a valid test number) is masked. The old demo used the invalid one.
- **`LLMToolEmulator` defaults to an Anthropic model.** Without `model=` it demands an Anthropic key, so
  always pass one explicitly.
- **`LLMToolSelectorMiddleware` runs before every model call**, not once per run: the tool list was filtered
  twice in a single two-step run.
- **`ToolErrorMiddleware` is opt-in.** Return a string to recover, return `None` to let the exception
  propagate (without any middleware, a raising tool simply crashes the run). Return the exception *type*,
  not `str(exc)`, so internals do not leak to the model.
- **`thread_limit` and HITL both need a checkpointer** (`InMemorySaver` + a `thread_id`); without one every
  `.invoke()` starts from nothing.
- **Free-tier reality (Groq):** roughly 8,000 tokens/min, 1,000 requests/day, plus per-model daily token
  budgets. `qwen/qwen3.8-27b` additionally has a 1,000 output-tokens/min limit and rejects requests that do
  not set `max_tokens`. `openai/gpt-oss-120b` is faster but occasionally emits a malformed tool call
  (`output_parse_failed`); `ModelRetryMiddleware` handles that.

## Model choice: Groq vs Gemini

Groq was used because a `GROQ_API_KEY` is already in `.env`, and no Gemini key is configured (the
`langchain-google-genai` package is installed in the venv, so adding a `GOOGLE_API_KEY` and changing
`make_model` would be enough to switch). Two Groq models were exercised:

| | `openai/gpt-oss-120b` | `qwen/qwen3.8-27b` |
|---|---|---|
| Saved outputs from | **this one** | earlier runs (all sections passed) |
| Speed | faster | slower |
| Tool calling | occasional malformed call / skipped call | very dependable, incl. forced tool calls |
| Free-tier friction | low | high (output-token cap, 200k tokens/day) |

## What changed when the two notebooks were merged

`Middleware.ipynb` (newer) was a superset of `MIddleware copy.ipynb` (older Colab run): nearly everything
in the older notebook was already in the newer one, which also added the Groq setup and the SQLite test
bench. So the newer file was the base, and the two (305 cells between them, most of it duplicated) became
**one 100-cell notebook** -- 71 cells of content plus 29 short "Observed in the saved run" notes --
reorganised into one section per middleware.

**Removed:** Colab `!pip install` / `userdata` cells; `print("hello")`; the unused `save_trip_demo` and
`inspect.getsource(...)` exploration; a note-to-self left in a code cell ("plz add comment for token...");
"paste it into Chat GPT" comments; stray `# BK1101` / `# 9th August` cells; a hard-coded personal email
address (replaced with `manager@example.com`); three repeated `from rich import print` cells (replaced by one
`show()` helper that prints a compact transcript instead of a 20-100 KB state dump).

**Fixed (these would not have run or did not test what they claimed):**
- `TodoListMiddleware(..., system_prompt=)` had no value -> `SyntaxError`.
- `!pip install` inside cells; `delay = ... ^ retry_number` and `1 * 2 ^ 0` (`^` is XOR, not power) -> a
  computed backoff table.
- `print(agent.invoke(...), config=config)` passed `config` to `print` instead of `invoke`.
- `run_interactive_hitl_demo` blocked on `input()` (cannot run unattended) and used the wrong `edit` shape.
- `trigger=('token', 4000)` (typo for `'tokens'`).
- The "real summarization" test had no checkpointer and `keep=2` (see above).
- The `ModelFallbackMiddleware` demo used OpenAI model names (including `gpt-5.5-haiku`, which is not a real
  model) -> Groq models with an *intentionally* nonexistent primary, plus a "without fallback" contrast.
- `list_bookings` printed all 40 rows to stdout on every call; it now returns them quietly.
- Markdown referring to "Part 2 / Part 3" and a `CineBotComplianceMiddleware` from a different course
  notebook that is not in this repo.

**Added:** the hook-lifecycle diagram and middleware table; a concept block per section; the
Gotchas above; compact, deterministic-ish output (`temperature=0`) and a from-scratch top-to-bottom run
with zero errors; `summary_prompt`, `respond`/`edit` HITL decisions, `exit_behavior="error"`, and the
retry-succeeds vs retry-exhausted contrast.

## Next steps

- **`ContextEditingMiddleware` + `ClearToolUsesEdit`**: prune old tool results instead of summarising them.
- **`ModelRetryMiddleware`** in its own section (backoff, `retry_on`, `on_failure`).
- **Middleware ordering experiments**, e.g. `ToolRetryMiddleware` inside `ToolErrorMiddleware`.
- **A persistent checkpointer** (SQLite/Postgres) instead of `InMemorySaver`, so a paused HITL run survives a
  restart.
- Other prebuilt middleware in `langchain 1.3.14`: `ShellToolMiddleware`, `FilesystemFileSearchMiddleware`.

> Custom decorator-style middleware (`before_model`/`after_model`, `wrap_model_call`), `Runtime`/
> `ToolRuntime`/`context_schema`, `dynamic_prompt`, and conditional (`when`-gated) HITL are now covered in
> `Runtime_Context_and_Custom_Middleware.ipynb` -- a full class-based `AgentMiddleware` example is still open.
