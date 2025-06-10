import pytest
from backend.collectors.producthunt import fetch_producthunt
from unittest.mock import patch, MagicMock
import os

def test_fetch_producthunt_returns_expected_structure():
    api_key = "dummy_api_key"
    data = fetch_producthunt(api_key)

    assert isinstance(data, list)
    assert len(data) > 0 # Ensure dummy data is not empty

    # Check structure of the first item
    first_item = data[0]
    assert "name" in first_item
    assert "description" in first_item
    assert "url" in first_item
    assert "category" in first_item
    assert "tags" in first_item
    assert "source" in first_item
    assert "collected_at" in first_item

    assert isinstance(first_item["name"], str)
    assert isinstance(first_item["description"], str)
    assert isinstance(first_item["url"], str)
    assert isinstance(first_item["category"], str)
    assert isinstance(first_item["tags"], list)
    assert all(isinstance(tag, str) for tag in first_item["tags"])
    assert isinstance(first_item["source"], str)
    
    # Check collected_at format (simple check for ISO format with Z)
    assert isinstance(first_item["collected_at"], str)
    assert first_item["collected_at"].endswith('Z')
    # You could add more robust datetime parsing and validation here if needed

# 향후 실제 API 연동 시, requests 모킹 예시:
# from unittest.mock import patch
# def test_fetch_producthunt_with_mocked_api_call():
#     with patch('requests.post') as mock_post:
#         mock_response = MagicMock()
#         mock_response.status_code = 200
#         mock_response.json.return_value = {"data": {"products": {"edges": []}}}
#         mock_post.return_value = mock_response
#         data = fetch_producthunt("real_api_key")
#         assert len(data) == 0
#         mock_post.assert_called_once() 