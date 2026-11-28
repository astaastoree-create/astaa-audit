"""Conversational data analyst: chat about a CSV file stored in Google Drive.

Fixed tools only - load, describe, filter, group-by aggregate. No arbitrary
code execution. The model picks the tool; the tools do exactly one thing each.
"""

from __future__ import annotations

import base64
import binascii
import csv
import io
import json
import os
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime

from mistralai import agents

try:  # mistralai >= 3 exports Mistral at the top level
    from mistralai import Mistral
except ImportError:  # pragma: no cover
    from mistralai.client import Mistral

DRIVE_CONNECTOR = "google_drive_mcp"

# ---------------------------------------------------------------------------
# Dataset model and per-session store
# ---------------------------------------------------------------------------


@dataclass
class Dataset:
    name: str
    file_id: str
    rows: list[dict[str, str]]
    columns: list[str]
    column_types: dict[str, str] = field(default_factory=dict)
    loaded_at: str = ""


_DATASETS: dict[str, Dataset] = {}
_MAX_KEY = 32


def _session_key(ctx: agents.ToolCallbackContext | None) -> str:
    if ctx is not None:
        session = getattr(ctx, "session", None)
        for attr in ("session_id", "id"):
            value = getattr(session, attr, None)
            if value:
                return str(value)[:_MAX_KEY]
    return "local"


def _get_dataset(ctx: agents.ToolCallbackContext | None) -> Dataset:
    dataset = _DATASETS.get(_session_key(ctx))
    if dataset is None:
        raise agents.ToolError(
            "No CSV is loaded in this conversation yet. "
            "Share the Google Drive link to the CSV file and I will load it with load_csv."
        )
    return dataset


# ---------------------------------------------------------------------------
# Type inference and coercion
# ---------------------------------------------------------------------------


_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$")
_DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S")


def _parse_bool(text: str) -> bool | None:
    lowered = text.strip().lower()
    if lowered in {"true", "yes", "y", "1"}:
        return True
    if lowered in {"false", "no", "n", "0"}:
        return False
    return None


def _parse_date(text: str) -> datetime | None:
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text.strip(), fmt)
        except ValueError:
            continue
    return None


def infer_type(values: list[str]) -> str:
    sample = [v for v in values if v is not None and str(v).strip() != ""]
    if not sample:
        return "text"
    if all(_INT_RE.match(str(v)) for v in sample):
        return "integer"
    if all(_FLOAT_RE.match(str(v)) for v in sample):
        return "number"
    if all(_parse_bool(str(v)) is not None for v in sample) and len({str(v).strip().lower() for v in sample}) <= 2:
        return "boolean"
    if all(_parse_date(str(v)) is not None for v in sample):
        return "date"
    return "text"


def _coerce(column: str, value: str, dataset: Dataset) -> str | float | bool | datetime:
    text = str(value)
    col_type = dataset.column_types.get(column, "text")
    if col_type in {"integer", "number"}:
        return float(text.replace(",", "").strip())
    if col_type == "boolean":
        parsed = _parse_bool(text)
        if parsed is None:
            raise agents.ToolError(f"Value '{value}' is not a valid boolean for column '{column}'.")
        return parsed
    if col_type == "date":
        parsed = _parse_date(text)
        if parsed is None:
            raise agents.ToolError(f"Value '{value}' is not a valid date for column '{column}' (expected e.g. 2024-01-31).")
        return parsed
    return text


# ---------------------------------------------------------------------------
# Analysis engine (pure functions, unit-testable)
# ---------------------------------------------------------------------------


def profile_column(dataset: Dataset, column: str) -> dict:
    if column not in dataset.columns:
        raise agents.ToolError(f"Column '{column}' does not exist. Columns: {dataset.columns}")
    values = [row[column] for row in dataset.rows]
    non_null = [v for v in values if v is not None and str(v).strip() != ""]
    col_type = dataset.column_types.get(column, "text")
    profile: dict = {
        "column": column,
        "type": col_type,
        "non_null": len(non_null),
        "missing": len(values) - len(non_null),
        "unique": len(set(non_null)),
    }
    if col_type in {"integer", "number"}:
        nums = [float(v.replace(",", "").strip()) for v in non_null]
        if nums:
            profile["min"] = _round(min(nums))
            profile["max"] = _round(max(nums))
            profile["mean"] = _round(statistics.fmean(nums))
            profile["median"] = _round(statistics.median(nums))
            profile["sum"] = _round(sum(nums))
    elif col_type == "date":
        dates = [_parse_date(v) for v in non_null]
        dates = [d for d in dates if d is not None]
        if dates:
            profile["min"] = min(dates).strftime("%Y-%m-%d")
            profile["max"] = max(dates).strftime("%Y-%m-%d")
    uniques = sorted({str(v) for v in non_null})
    if len(uniques) <= 15:
        profile["values"] = uniques
    else:
        counts: dict[str, int] = {}
        for v in non_null:
            counts[str(v)] = counts.get(str(v), 0) + 1
        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
        profile["top_values"] = [{"value": v, "count": c} for v, c in top]
    return profile


_FILTER_OPS = {
    "eq", "ne", "gt", "gte", "lt", "lte",
    "contains", "starts_with", "ends_with", "in",
    "is_null", "not_null", "between",
}


def _matches(row: dict[str, str], column: str, op: str, value: object, value2: object, dataset: Dataset) -> bool:
    raw = row.get(column)
    is_null = raw is None or str(raw).strip() == ""
    if op == "is_null":
        return is_null
    if op == "not_null":
        return not is_null
    if is_null:
        return False
    if op == "contains":
        return str(value).lower() in str(raw).lower()
    if op == "starts_with":
        return str(raw).lower().startswith(str(value).lower())
    if op == "ends_with":
        return str(raw).lower().endswith(str(value).lower())
    if op == "in":
        items = value if isinstance(value, list) else str(value).split("|")
        return str(raw).strip() in {str(item).strip() for item in items}
    left = _coerce(column, str(raw), dataset)
    if op == "eq":
        return _equal(left, _coerce(column, str(value), dataset))
    if op == "ne":
        return not _equal(left, _coerce(column, str(value), dataset))
    right = _coerce(column, str(value), dataset)
    if op == "gt":
        return left > right  # type: ignore[operator]
    if op == "gte":
        return left >= right  # type: ignore[operator]
    if op == "lt":
        return left < right  # type: ignore[operator]
    if op == "lte":
        return left <= right  # type: ignore[operator]
    if op == "between":
        if value2 is None:
            raise agents.ToolError("Op 'between' needs value2 as the upper bound.")
        upper = _coerce(column, str(value2), dataset)
        return right <= left <= upper  # type: ignore[operator]
    raise agents.ToolError(f"Unknown op '{op}'. Valid ops: {sorted(_FILTER_OPS)}")


def _equal(left: object, right: object) -> bool:
    if isinstance(left, float) and isinstance(right, (int, float)):
        return abs(left - float(right)) < 1e-9
    return left == right


def filter_dataset(
    dataset: Dataset,
    column: str,
    op: str,
    value: object = None,
    value2: object = None,
) -> list[dict[str, str]]:
    if column not in dataset.columns:
        raise agents.ToolError(f"Column '{column}' does not exist. Columns: {dataset.columns}")
    if op not in _FILTER_OPS:
        raise agents.ToolError(f"Unknown op '{op}'. Valid ops: {sorted(_FILTER_OPS)}")
    return [row for row in dataset.rows if _matches(row, column, op, value, value2, dataset)]


_AGGREGATIONS = {
    "count", "count_distinct", "sum", "mean", "avg", "min", "max", "median",
}


def aggregate_dataset(
    dataset: Dataset,
    group_by: list[str],
    metrics: dict[str, str],
    rows: list[dict[str, str]] | None = None,
    order_by: str | None = None,
    limit: int = 50,
) -> dict:
    for column in group_by:
        if column not in dataset.columns:
            raise agents.ToolError(f"Group-by column '{column}' does not exist. Columns: {dataset.columns}")
    if not metrics:
        raise agents.ToolError("metrics is empty. Pass at least one, e.g. {\"revenue\": \"sum\"}.")
    for column, agg in metrics.items():
        if column not in dataset.columns:
            raise agents.ToolError(f"Metric column '{column}' does not exist. Columns: {dataset.columns}")
        if agg not in _AGGREGATIONS:
            raise agents.ToolError(f"Unknown aggregation '{agg}' for column '{column}'. Valid: {sorted(_AGGREGATIONS)}")

    source = dataset.rows if rows is None else rows
    groups: dict[tuple, list[dict[str, str]]] = {}
    for row in source:
        key = tuple("" if row.get(c) is None else str(row.get(c)) for c in group_by)
        groups.setdefault(key, []).append(row)

    result_rows = []
    for key, members in groups.items():
        out: dict[str, object] = {c: v for c, v in zip(group_by, key)}
        for column, agg in metrics.items():
            out[f"{agg}({column})"] = _round(_aggregate_column(members, column, agg))
        result_rows.append(out)

    if order_by:
        if order_by not in result_rows[0]:
            raise agents.ToolError(f"order_by '{order_by}' is not a result column: {list(result_rows[0])}")
        result_rows.sort(key=lambda r: _sort_key(r[order_by]), reverse=True)

    total = len(result_rows)
    truncated = total > limit
    return {"groups": result_rows[:limit], "group_count": total, "truncated": truncated}


def _aggregate_column(rows: list[dict[str, str]], column: str, agg: str) -> float:
    values = [row.get(column) for row in rows]
    non_null = [v for v in values if v is not None and str(v).strip() != ""]
    if agg == "count":
        return float(len(non_null))
    if agg == "count_distinct":
        return float(len({str(v) for v in non_null}))
    nums: list[float] = []
    for v in non_null:
        text = str(v).replace(",", "").strip()
        try:
            nums.append(float(text))
        except ValueError:
            continue
    if not nums:
        return 0.0
    if agg == "sum":
        return sum(nums)
    if agg in {"mean", "avg"}:
        return statistics.fmean(nums)
    if agg == "median":
        return statistics.median(nums)
    if agg == "min":
        return min(nums)
    if agg == "max":
        return max(nums)
    raise agents.ToolError(f"Unknown aggregation '{agg}'.")


def _round(value: float) -> float:
    return round(value, 6)


def _sort_key(value: object) -> tuple[int, object]:
    if isinstance(value, bool):
        return (0, value)
    if isinstance(value, (int, float)):
        return (0, value)
    return (1, str(value))


# ---------------------------------------------------------------------------
# Google Drive loading (direct connector tool call, no model in the loop)
# ---------------------------------------------------------------------------

_DRIVE_ID_PATTERNS = [
    re.compile(r"drive\.google\.com/file/d/([A-Za-z0-9_-]{10,})"),
    re.compile(r"docs\.google\.com/spreadsheets/d/([A-Za-z0-9_-]{10,})"),
    re.compile(r"docs\.google\.com/document/d/([A-Za-z0-9_-]{10,})"),
    re.compile(r"[?&]id=([A-Za-z0-9_-]{10,})"),
    re.compile(r"/d/([A-Za-z0-9_-]{10,})"),
]


def extract_drive_file_id(link: str) -> str:
    link = link.strip()
    for pattern in _DRIVE_ID_PATTERNS:
        match = pattern.search(link)
        if match:
            return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", link):
        return link
    raise agents.ToolError(
        "Could not find a Google Drive file ID in that link. "
        "Use a link like https://drive.google.com/file/d/<FILE_ID>/view or ?id=<FILE_ID>."
    )


def _mistral_client() -> Mistral:
    api_key = os.environ.get("MISTRAL_API_KEY") or os.environ.get("MISTRAL_CLIENT_API_KEY")
    if not api_key:
        raise agents.ToolError(
            "MISTRAL_API_KEY is not set in the agent's environment, so Google Drive cannot be reached."
        )
    return Mistral(api_key=api_key)


def _result_text(result: object) -> str:
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        payload = structured.get("content")
        if isinstance(payload, str) and payload:
            return payload
    content = getattr(result, "content", None) or []
    parts = []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts)


def _looks_like_csv(text: str) -> bool:
    head = text.lstrip("\ufeff")[:4096]
    return ("," in head or "\t" in head or ";" in head) and "\n" in head


def _decode_payload(raw: str) -> str:
    text = raw.strip()
    if text.startswith("{"):
        try:
            payload = json.loads(text).get("content")
            if isinstance(payload, str) and payload:
                text = payload
        except json.JSONDecodeError:
            pass
    if not _looks_like_csv(text):
        try:
            decoded = base64.b64decode(text, validate=True).decode("utf-8-sig")
            if _looks_like_csv(decoded):
                return decoded
        except (binascii.Error, UnicodeDecodeError, ValueError):
            pass
    return text


async def fetch_drive_csv(file_id: str) -> tuple[str, str]:
    client = _mistral_client()
    result = await client.beta.connectors.call_tool_async(
        connector_id_or_name=DRIVE_CONNECTOR,
        tool_name=DRIVE_DOWNLOAD_TOOL,
        arguments={"file_id": file_id, "export_mime_type": "text/csv"},
    )
    raw = _result_text(result)
    if not raw:
        raise agents.ToolError(f"Google Drive returned no content for file {file_id}.")
    csv_text = _decode_payload(raw)
    title = ""
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        title = str(structured.get("title") or "")
    if not title and raw.strip().startswith("{"):
        try:
            title = str(json.loads(raw.strip()).get("title") or "")
        except json.JSONDecodeError:
            pass
    return csv_text, title


# ---------------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------------


@agents.tool
async def load_csv(ctx: agents.ToolCallbackContext, /, file_link: str) -> dict:
    """Load a CSV file from Google Drive into this conversation.

    Pass the Google Drive link (or file ID) the user shared. The file is
    downloaded through the user's Google Drive connection; the CSV is parsed
    and kept for the follow-up tools. Returns the file name, row count and
    the columns with their detected types.
    """
    file_id = extract_drive_file_id(file_link)
    csv_text, file_title = await fetch_drive_csv(file_id)
    reader = csv.DictReader(io.StringIO(csv_text))
    if not reader.fieldnames:
        raise agents.ToolError("The file does not look like a CSV with a header row.")
    rows = []
    for raw_row in reader:
        row = {k: ("" if v is None else str(v)) for k, v in raw_row.items() if k is not None}
        rows.append(row)
    if not rows:
        raise agents.ToolError("The CSV has a header row but no data rows.")
    columns = list(reader.fieldnames)
    column_types = {c: infer_type([r[c] for r in rows]) for c in columns}
    dataset = Dataset(
        name=file_title or f"google-drive:{file_id}",
        file_id=file_id,
        rows=rows,
        columns=columns,
        column_types=column_types,
        loaded_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )
    _DATASETS[_session_key(ctx)] = dataset
    preview = [{c: r[c] for c in columns[:8]} for r in rows[:3]]
    return {
        "file_id": file_id,
        "row_count": len(rows),
        "columns": [{"name": c, "type": column_types[c]} for c in columns],
        "preview": preview,
    }


@agents.tool
async def describe_columns(ctx: agents.ToolCallbackContext, /, column: str | None = None) -> dict:
    """Describe the loaded CSV's columns: type, missing values, unique count, and
    for numeric columns min/max/mean/median/sum, for date columns the range,
    for text columns the distinct values or the top 5 most frequent."""
    dataset = _get_dataset(ctx)
    if column:
        return {"columns": [profile_column(dataset, column)]}
    return {
        "file_id": dataset.file_id,
        "row_count": len(dataset.rows),
        "columns": [profile_column(dataset, c) for c in dataset.columns],
    }


@agents.tool
async def filter_rows(
    ctx: agents.ToolCallbackContext,
    /,
    column: str,
    op: str,
    value: object = None,
    value2: object = None,
    limit: int = 20,
) -> dict:
    """Return the rows of the loaded CSV that match one condition.

    op is one of: eq, ne, gt, gte, lt, lte, contains, starts_with, ends_with,
    in (value as "a|b|c"), between (value and value2), is_null, not_null.
    Value is compared in the column's detected type. Returns the match count
    and up to `limit` example rows.
    """
    dataset = _get_dataset(ctx)
    matches = filter_dataset(dataset, column, op, value, value2)
    limit = max(1, min(int(limit), 50))
    truncated = len(matches) > limit
    return {
        "match_count": len(matches),
        "truncated": truncated,
        "rows": matches[:limit],
    }


@agents.tool
async def group_by_aggregate(
    ctx: agents.ToolCallbackContext,
    /,
    group_by: list[str],
    metrics: dict[str, str],
    filter_column: str | None = None,
    filter_op: str | None = None,
    filter_value: object = None,
    filter_value2: object = None,
    order_by: str | None = None,
    limit: int = 50,
) -> dict:
    """Group the loaded CSV by one or more columns and compute aggregates.

    metrics maps a column to an aggregation: count, count_distinct, sum, mean,
    avg, median, min or max, e.g. {"revenue": "sum", "order_id": "count"}.
    Optionally pre-filter with one condition (filter_column + filter_op and
    filter_value / filter_value2, same ops as filter_rows). order_by sorts by
    a result column, descending. Returns the group table.
    """
    dataset = _get_dataset(ctx)
    rows = dataset.rows
    if filter_column:
        rows = filter_dataset(dataset, filter_column, str(filter_op), filter_value, filter_value2)
    return aggregate_dataset(dataset, list(group_by), dict(metrics), rows=rows, order_by=order_by, limit=max(1, min(int(limit), 200)))


# Confirmed against the google_drive_mcp connector's tool list.
DRIVE_DOWNLOAD_TOOL = "download_file_content"


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

INSTRUCTIONS = """You are a conversational data analyst. The user shares a Google Drive link to a CSV file and asks questions about it in plain language.

How you work:
- When the user gives a Drive link (or asks about "the file" and none is loaded), call load_csv with the link first.
- Answer every data question with numbers produced by your tools: describe_columns to profile columns, filter_rows to count or show matching rows, group_by_aggregate for group-level sums, averages, medians, min/max and counts.
- Never invent or estimate numbers. If your tools cannot answer a question with the loaded data, say so plainly.
- The loaded file stays available for the whole conversation: never ask the user to re-share it, and do not call load_csv again for the same link unless they ask.
- Combine tools when needed: filter first, then group the filtered rows (group_by_aggregate accepts filter_column/filter_op/filter_value).
- Before computing on a column you have not profiled yet, a quick describe_columns call keeps types straight.
- Present results concisely: lead with the answer, give the key figures, use a small table for group comparisons, and mention how many rows or groups a number covers.
- The user speaks plainly; mirror their language and keep the tone direct."""


agent = agents.Agent(
    model="mistral-medium-latest",
    name="csv-analyst",
    instructions=INSTRUCTIONS,
    tools=[load_csv, describe_columns, filter_rows, group_by_aggregate],
)
