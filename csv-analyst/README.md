# csv-analyst

A conversational data analyst agent built with the [Mistral Agents SDK](https://docs.mistral.ai/studio/agents/getting-started/quickstart).
You share a Google Drive link to a CSV file and ask questions in plain language;
it answers with real numbers from the data.

## Design

- **Fixed tools, no code execution.** The model can only call these four tools:

  | Tool | What it does |
  | --- | --- |
  | `load_csv(file_link)` | Resolves the Drive file ID, downloads the file through your Google Drive connection (direct connector tool call — no model in the loop), parses the CSV, and stores it per session. |
  | `describe_columns(column?)` | Per column: type, missing/unique counts, min/max/mean/median/sum (numeric), date range (dates), distinct or top-5 values (text). |
  | `filter_rows(column, op, value, ...)` | Count and sample rows matching one condition: `eq, ne, gt, gte, lt, lte, contains, starts_with, ends_with, in, between, is_null, not_null`. |
  | `group_by_aggregate(group_by, metrics, ...)` | Group by one or more columns, compute `count, count_distinct, sum, mean, avg, median, min, max`, with an optional pre-filter and ordering. |

- **Session memory.** The loaded file is stored keyed by session ID, and the
  session keeps the whole conversation history, so follow-ups work without
  re-sharing the file.
- **No arbitrary code execution.** No code interpreter, no shell; the model
  only picks among the four typed tools above.

## Prerequisites

1. A Google Drive connector connected on your Mistral account (Studio → Context → Connectors).
2. A Mistral API key with Workflows access (Developer or Workspace Contributor role), in `.env` as `MISTRAL_API_KEY`.
3. For deployment: a workspace secret holding that key (console.mistral.ai → Secrets), e.g. `mistral-api-key`.

## Run locally

```bash
uv sync
uv run --env-file .env agents chat
```

## Tests

```bash
uv run pytest
```

## Deploy (Mistral-managed)

```bash
git add --all && git commit -m "csv-analyst agent" && git push
uv run --env-file .env agents deploy --secret-binding MISTRAL_API_KEY=secret:workspace:mistral-api-key
```

Then chat with it in Vibe Code Web at `https://chat.mistral.ai/managed-agents/csv-analyst`,
or from the terminal with `uv run --env-file .env agents chat csv-analyst`.
