import pytest
from unittest.mock import patch, MagicMock
import os
import json
import pandas as pd
import faiss
import numpy as np
import tempfile

# Ensure these imports are correct based on your project structure
from backend.indexer.indexer import index_data, build_faiss_index, save_faiss_index, load_faiss_index
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.schema import Document

@patch('os.path.exists')
@patch('builtins.open', new_callable=MagicMock)
@patch('json.load')
@patch('llama_index.core.vector_stores.simple.SimpleVectorStore')
@patch('llama_index.core.storage_context.StorageContext.from_defaults')
@patch('llama_index.embeddings.huggingface.HuggingFaceEmbedding')
@patch('llama_index.core.VectorStoreIndex.from_documents')
def test_index_data_success(mock_from_documents, mock_embedding, mock_storage_context_from_defaults,
                            mock_simple_vector_store, mock_json_load, mock_open, mock_exists):
    # Setup mocks for successful scenario
    mock_exists.return_value = True # processed_chunks.json exists
    mock_json_load.return_value = [
        {"id": "1", "text": "test chunk 1", "metadata": {"source": "test"}}
    ]
    mock_vector_store_instance = MagicMock(spec=SimpleVectorStore)
    mock_simple_vector_store.return_value = mock_vector_store_instance
    mock_storage_context_instance = MagicMock()
    mock_storage_context_from_defaults.return_value = mock_storage_context_instance

    index_data()

    # Assertions
    mock_exists.assert_called_with(os.path.join("processed_data_storage", "processed_chunks.json"))
    mock_json_load.assert_called_once()
    mock_embedding.assert_called_once_with(model_name="thenlper/gte-large", device="cpu")
    mock_from_documents.assert_called_once()
    mock_storage_context_instance.persist.assert_called_once()

@patch('os.path.exists')
@patch('builtins.print')
def test_index_data_no_chunks_file(mock_print, mock_exists):
    mock_exists.return_value = False # processed_chunks.json does not exist
    index_data()
    mock_print.assert_any_call('오류: 처리된 청크 파일이 없습니다. processed_data_storage\\processed_chunks.json를 확인해주세요.')

# Additional scenarios: empty chunks file, loading existing vector store, etc.

@patch('sentence_transformers.SentenceTransformer')
def test_build_faiss_index_success(mock_sentence_transformer):
    mock_model_instance = MagicMock()
    mock_model_instance.encode.return_value = np.array([[0.1] * 384, [0.2] * 384], dtype='float32')
    mock_sentence_transformer.return_value = mock_model_instance

    dummy_df = pd.DataFrame({
        'description': ['text one', 'text two'],
        'other_col': ['a', 'b']
    })
    index = build_faiss_index(dummy_df)

    assert isinstance(index, faiss.IndexFlatIP)
    assert index.ntotal == 2
    assert index.d == 384
    mock_model_instance.encode.assert_called_once()

def test_build_faiss_index_nan_description():
    mock_model_instance = MagicMock()
    mock_model_instance.encode.return_value = np.array([[0.1] * 384], dtype='float32')
    with patch('sentence_transformers.SentenceTransformer', return_value=mock_model_instance):
        dummy_df = pd.DataFrame({
            'description': ['text one', np.nan],
            'other_col': ['a', 'b']
        })
        index = build_faiss_index(dummy_df)
        assert index.ntotal == 2 # NaN should be treated as empty string, still encoded
        mock_model_instance.encode.assert_called_once_with(['text one', ''], show_progress_bar=True)

def test_save_and_load_faiss_index():
    # Create a dummy FAISS index
    d = 64
    nb = 10
    xb = np.random.rand(nb, d).astype('float32')
    original_index = faiss.IndexFlatIP(d)
    original_index.add(xb)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tmp_file:
        test_path = tmp_file.name

    try:
        save_faiss_index(original_index, test_path)
        assert os.path.exists(test_path)

        loaded_index = load_faiss_index(test_path)
        assert loaded_index.ntotal == original_index.ntotal
        assert loaded_index.d == original_index.d
        # More robust comparison: check if vectors are the same
        # D, I = loaded_index.search(xb[:1], 1) 
        # assert I[0][0] == 0

    finally:
        if os.path.exists(test_path):
            os.remove(test_path) 