import pytest
from unittest.mock import patch, MagicMock, ANY
import os
import json
import tempfile
import numpy as np

# Ensure these imports are correct based on your project structure
from backend.parser.parser import extract_metadata, parse_and_chunk, embedding_model
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode

def test_extract_metadata_price():
    text = "Our plan starts at $19.99/month. Includes API."
    metadata = extract_metadata(text)
    assert metadata["price"] == 19.99
    assert metadata["has_api"] == True

def test_extract_metadata_categories_and_tags():
    text = "Generate stunning images with our AI art generator. Also great for video editing."
    metadata = extract_metadata(text)
    assert "Image Generation" in metadata["category"]
    assert "Video Editing" in metadata["category"]
    assert "image" in metadata["tags"]
    assert "art" in metadata["tags"]
    assert "video" in metadata["tags"]
    assert len(set(metadata["tags"])) == len(metadata["tags"]) # Check for no duplicates

def test_extract_metadata_no_match():
    text = "A generic tool for daily tasks."
    metadata = extract_metadata(text)
    assert metadata["price"] is None
    assert metadata["has_api"] == False
    assert metadata["category"] == []
    assert metadata["tags"] == []

@pytest.fixture
def setup_temp_dirs():
    with tempfile.TemporaryDirectory() as raw_dir, \
         tempfile.TemporaryDirectory() as processed_dir:
        # Mock os.path.dirname(__file__) to point to a test-controlled path
        with patch('os.path.dirname', return_value=os.path.join(os.getcwd(), "backend/parser")) as mock_dirname:
            # Mock the paths used in parse_and_chunk to use temp dirs
            with patch('backend.parser.parser.raw_html_dir', raw_dir), \
                 patch('backend.parser.parser.processed_data_dir', processed_dir):
                yield raw_dir, processed_dir

@patch('backend.parser.parser.SentenceTransformer') # Mock the SentenceTransformer loading
@patch('backend.parser.parser.faiss.write_index')
@patch('backend.parser.parser.os.path.exists') # Mock os.path.exists
@patch('backend.parser.parser.os.listdir') # Mock os.listdir
@patch('builtins.open', new_callable=MagicMock) # Mock open
def test_parse_and_chunk_success(mock_open, mock_listdir, mock_exists, mock_write_index, mock_sentence_transformer, setup_temp_dirs):
    raw_dir, processed_dir = setup_temp_dirs

    # Configure mocks
    # We need to simulate that raw_dir exists and the specific HTML file exists within it.
    # For a mocked os.path.exists, we can define its behavior for specific paths.
    def mock_exists_side_effect(path):
        if path == raw_dir:
            return True
        if path == os.path.join(raw_dir, "test_ai_tool.html"):
            return True
        if path == processed_dir: # Ensure processed_dir also exists
            return True
        return False
    mock_exists.side_effect = mock_exists_side_effect
    
    mock_listdir.return_value = ["test_ai_tool.html"] # Simulate a file in raw_dir

    # Mock the open call to return the content directly
    mock_file = MagicMock()
    mock_file.__enter__.return_value.read.return_value = "<html><head><title>Test AI Tool</title></head><body><h1>Welcome</h1><p>This is an AI tool for <b>image generation</b>.</p></body></html>"
    mock_open.return_value = mock_file

    # Mock embedding model to return dummy embeddings
    mock_embed_instance = MagicMock()
    mock_embed_instance.encode.return_value = np.array([[0.1] * 384], dtype='float32') # Use 384 dimensions
    mock_sentence_transformer.return_value = mock_embed_instance

    parse_and_chunk()

    # Assertions
    mock_exists.assert_any_call(raw_dir)
    mock_listdir.assert_called_once_with(raw_dir)
    mock_open.assert_any_call(os.path.join(raw_dir, "test_ai_tool.html"), 'r', encoding='utf-8')
    mock_write_index.assert_called_once_with(ANY, os.path.join(processed_dir, "ai_tools_faiss_index.bin"))
    mock_open.assert_any_call(os.path.join(processed_dir, "processed_ai_tools_metadata.json"), 'w', encoding='utf-8')

    with open(os.path.join(processed_dir, "processed_ai_tools_metadata.json"), 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    assert len(metadata) > 0
    assert metadata[0]["name"] == "Test AI Tool"
    assert "image generation" in metadata[0]["description"].lower()
    assert "Image Generation" in metadata[0]["category"]
    assert "image" in metadata[0]["tags"]
    assert "ai" in metadata[0]["tags"]

@patch('backend.parser.parser.faiss.write_index')
@patch('builtins.print')
@patch('backend.parser.parser.os.path.exists') # Mock os.path.exists
@patch('backend.parser.parser.os.listdir') # Mock os.listdir
def test_parse_and_chunk_no_html_files(mock_listdir, mock_exists, mock_print, mock_write_index, setup_temp_dirs):
    raw_dir, processed_dir = setup_temp_dirs
    
    mock_exists.return_value = True # Simulate raw_dir exists
    mock_listdir.return_value = [] # No HTML files

    parse_and_chunk()

    mock_print.assert_any_call(f'오류: 원본 HTML 파일이 없습니다. {raw_dir} 디렉토리를 확인해주세요.')
    mock_write_index.assert_not_called() # No index should be written if no files

    assert not os.path.exists(os.path.join(processed_dir, "ai_tools_faiss_index.bin"))
    assert not os.path.exists(os.path.join(processed_dir, "processed_ai_tools_metadata.json")) 