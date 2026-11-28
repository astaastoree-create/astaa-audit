"""Unit tests for the analysis engine and link parsing (no network needed)."""

import pytest

import main


@pytest.fixture()
def dataset() -> main.Dataset:
    rows = [
        {"order_id": "1", "region": "north", "revenue": "120.50", "units": "3", "when": "2024-01-05", "vip": "true", "note": ""},
        {"order_id": "2", "region": "south", "revenue": "80", "units": "1", "when": "2024-01-09", "vip": "false", "note": "gift"},
        {"order_id": "3", "region": "north", "revenue": "200", "units": "5", "when": "2024-02-01", "vip": "true", "note": ""},
        {"order_id": "4", "region": "east", "revenue": "60.25", "units": "2", "when": "2024-02-11", "vip": "false", "note": ""},
        {"order_id": "5", "region": "south", "revenue": "", "units": "", "when": "", "vip": "", "note": ""},
    ]
    columns = list(rows[0])
    types = {c: main.infer_type([r[c] for r in rows]) for c in columns}
    return main.Dataset(name="t", file_id="f", rows=rows, columns=columns, column_types=types)


def test_infer_type():
    assert main.infer_type(["1", "2", "3"]) == "integer"
    assert main.infer_type(["1.5", "2"]) == "number"
    assert main.infer_type(["2024-01-02", "2024-02-03"]) == "date"
    assert main.infer_type(["true", "false"]) == "boolean"
    assert main.infer_type(["a", "b"]) == "text"


def test_extract_drive_file_id():
    assert main.extract_drive_file_id("https://drive.google.com/file/d/1AbC_def-1234567890/view") == "1AbC_def-1234567890"
    assert main.extract_drive_file_id("https://drive.google.com/open?id=1AbC_def-1234567890") == "1AbC_def-1234567890"
    assert main.extract_drive_file_id("https://docs.google.com/spreadsheets/d/1AbC_def-1234567890/edit") == "1AbC_def-1234567890"
    assert main.extract_drive_file_id("1AbC_def-12345678901") == "1AbC_def-12345678901"
    with pytest.raises(main.agents.ToolError):
        main.extract_drive_file_id("https://example.com/nothing")


def test_filter_numeric(dataset):
    rows = main.filter_dataset(dataset, "revenue", "gte", "100")
    assert [r["order_id"] for r in rows] == ["1", "3"]


def test_filter_text_and_null(dataset):
    assert [r["order_id"] for r in main.filter_dataset(dataset, "region", "eq", "south")] == ["2", "5"]
    assert [r["order_id"] for r in main.filter_dataset(dataset, "note", "is_null")] == ["1", "3", "4", "5"]
    assert [r["order_id"] for r in main.filter_dataset(dataset, "region", "in", "north|east")] == ["1", "3", "4"]


def test_filter_between_and_unknown_op(dataset):
    rows = main.filter_dataset(dataset, "revenue", "between", "60.25", "120.50")
    assert [r["order_id"] for r in rows] == ["1", "2", "4"]
    with pytest.raises(main.agents.ToolError):
        main.filter_dataset(dataset, "region", "frobnicate", "x")


def test_aggregate_group_by(dataset):
    result = main.aggregate_dataset(
        dataset, ["region"], {"revenue": "sum", "order_id": "count"}, order_by="sum(revenue)"
    )
    by_region = {g["region"]: g for g in result["groups"]}
    assert by_region["north"]["sum(revenue)"] == 320.5
    assert by_region["north"]["count(order_id)"] == 2
    assert by_region["south"]["sum(revenue)"] == 80.0
    assert by_region["south"]["count(order_id)"] == 2
    assert result["group_count"] == 3


def test_aggregate_with_prefilter(dataset):
    rows = main.filter_dataset(dataset, "revenue", "gt", "70")
    result = main.aggregate_dataset(dataset, ["region"], {"revenue": "mean"}, rows=rows)
    assert result["group_count"] == 2
    by_region = {g["region"]: g for g in result["groups"]}
    assert by_region["north"]["mean(revenue)"] == 160.25


def test_aggregate_median_min_max(dataset):
    result = main.aggregate_dataset(dataset, [], {"revenue": "median", "units": "max"})
    assert result["groups"][0]["median(revenue)"] == 100.25
    assert result["groups"][0]["max(units)"] == 5.0


def test_aggregate_unknown_metric(dataset):
    with pytest.raises(main.agents.ToolError):
        main.aggregate_dataset(dataset, ["region"], {"revenue": "variance"})


def test_profile_column(dataset):
    revenue = main.profile_column(dataset, "revenue")
    assert revenue["type"] == "number"
    assert revenue["missing"] == 1
    assert revenue["sum"] == 460.75
    region = main.profile_column(dataset, "region")
    assert sorted(region["values"]) == ["east", "north", "south"]
    with pytest.raises(main.agents.ToolError):
        main.profile_column(dataset, "nope")


def test_agent_has_fixed_tools():
    names = sorted(
        t.tool_definition.name for t in (main.load_csv, main.describe_columns, main.filter_rows, main.group_by_aggregate)
    )
    assert names == ["describe_columns", "filter_rows", "group_by_aggregate", "load_csv"]
    for t in (main.load_csv, main.describe_columns, main.filter_rows, main.group_by_aggregate):
        assert t.tool_definition.source is None
