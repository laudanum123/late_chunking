"""Tests for embedder implementations."""
import pytest
import numpy as np
from pathlib import Path
import os
from unittest.mock import Mock, patch

from late_chunking.embedders import (
    EmbeddingConfig,
    HuggingFaceEmbedder,
    OpenAIEmbedder,
    ModelLoadError,
    EmbeddingProcessError
)

@pytest.fixture
def hf_config():
    """Fixture for HuggingFace embedder config."""
    return EmbeddingConfig(
        name="jinaai/jina-embeddings-v2-base-en",
        type="huggingface",
        embedding_size=768,
        max_length=8192
    )

@pytest.fixture
def openai_config():
    """Fixture for OpenAI embedder config."""
    return EmbeddingConfig(
        name="text-embedding-ada-002",
        type="openai",
        embedding_size=1536,
        max_length=8191,
        api_key="test-key"
    )

@pytest.mark.asyncio
async def test_huggingface_embedder_initialization(hf_config):
    """Test HuggingFace embedder initialization."""
    with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
        with patch('transformers.AutoModel.from_pretrained') as mock_model:
            embedder = HuggingFaceEmbedder(hf_config)
            assert embedder.config == hf_config
            mock_tokenizer.assert_called_once()
            mock_model.assert_called_once()

@pytest.mark.asyncio
async def test_openai_embedder_initialization(openai_config):
    """Test OpenAI embedder initialization."""
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
        embedder = OpenAIEmbedder(openai_config)
        assert embedder.config == openai_config

@pytest.mark.asyncio
async def test_huggingface_embed_chunks(hf_config):
    """Test HuggingFace embedder chunk embedding."""
    with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
        with patch('transformers.AutoModel.from_pretrained') as mock_model:
            mock_output = Mock()
            mock_output.last_hidden_state = np.random.randn(1, 1, 768)
            mock_model.return_value.return_value = mock_output
            
            embedder = HuggingFaceEmbedder(hf_config)
            chunks = ["Test chunk 1", "Test chunk 2"]
            embeddings = await embedder.embed_chunks(chunks)
            
            assert isinstance(embeddings, np.ndarray)
            assert embeddings.shape[1] == hf_config.embedding_size

@pytest.mark.asyncio
async def test_openai_embed_chunks(openai_config):
    """Test OpenAI embedder chunk embedding."""
    mock_response = {
        'data': [{
            'embedding': np.random.randn(openai_config.embedding_size).tolist()
        }]
    }
    
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
        with patch('openai.Embedding.acreate', return_value=mock_response):
            embedder = OpenAIEmbedder(openai_config)
            chunks = ["Test chunk"]
            embeddings = await embedder.embed_chunks(chunks)
            
            assert isinstance(embeddings, np.ndarray)
            assert embeddings.shape[1] == openai_config.embedding_size

@pytest.mark.asyncio
async def test_invalid_chunks():
    """Test handling of invalid chunks."""
    embedder = OpenAIEmbedder(openai_config)
    with pytest.raises(ValueError):
        await embedder.embed_chunks([])
    with pytest.raises(ValueError):
        await embedder.embed_chunks([1, 2, 3])  # Non-string chunks
