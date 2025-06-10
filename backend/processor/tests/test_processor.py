import pytest
import pandas as pd
from backend.processor.processor import clean_text, preprocess

def test_clean_text_html_tags():
    text = "<p>Hello <b>World</b>!<br/>How are you?</p>"
    expected = "hello world! how are you?"
    assert clean_text(text) == expected

def test_clean_text_whitespace():
    text = "  This   is\n a \t test.  "
    expected = "this is a test."
    assert clean_text(text) == expected

def test_clean_text_non_string_input():
    assert clean_text(None) == ""
    assert clean_text(123) == ""

def test_preprocess_basic_cleaning_and_deduplication():
    records = [
        {
            "name": "Tool A",
            "description": "<p>Desc <b>one</b>.</p>",
            "url": "http://example.com/a",
            "source": "test",
            "category": "Utils", "tags": ["t1"], "collected_at": "2024-01-01"
        },
        {
            "name": "Tool B",
            "description": " Desc Two  ",
            "url": "http://example.com/b",
            "source": "test",
            "category": "Dev", "tags": ["t2"], "collected_at": "2024-01-02"
        },
        {
            "name": "Tool A", # Duplicate
            "description": "Another desc one.",
            "url": "http://example.com/a",
            "source": "test",
            "category": "Utils", "tags": ["t1"], "collected_at": "2024-01-03"
        },
        {
            "name": "Tool C",
            "description": None, # Missing required field
            "url": "http://example.com/c",
            "source": "test",
            "category": "Other", "tags": ["t3"], "collected_at": "2024-01-04"
        }
    ]
    df = preprocess(records)

    # Check deduplication and missing fields removal
    assert len(df) == 2
    assert "Tool A" in df["name"].tolist()
    assert "Tool B" in df["name"].tolist()
    assert "Tool C" not in df["name"].tolist()

    # Check text cleaning
    assert df[df["name"] == "Tool A"]["description"].iloc[0] == "desc one."
    assert df[df["name"] == "Tool B"]["description"].iloc[0] == "desc two"

    # Check required fields presence
    expected_cols = ["name", "description", "url", "category", "tags", "source", "collected_at"]
    assert all(col in df.columns for col in expected_cols)

def test_preprocess_empty_records():
    df = preprocess([])
    assert df.empty
    expected_cols = ["name", "description", "url", "category", "tags", "source", "collected_at"]
    assert all(col in df.columns for col in expected_cols) # Even for empty, schema should be consistent 