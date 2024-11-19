"""Tests for embedder implementations."""
import pytest
import numpy as np
from pathlib import Path
import os
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import torch
import tempfile
import shutil
from transformers import BatchEncoding

from late_chunking.embedders import (
    EmbeddingConfig,
    HuggingFaceEmbedder,
    OpenAIEmbedder,
    ModelLoadError,
    EmbeddingProcessError,
    ChunkWithEmbedding,
    LateChunkingEmbedder
)
from late_chunking.chunkers import TokenizerBasedSentenceChunker, ChunkMetadata

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

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
    mock_tokenizer = Mock()
    mock_model = Mock()
    mock_model.to = Mock(return_value=mock_model)
    
    with patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer) as mock_tokenizer_cls:
        with patch('transformers.AutoModel.from_pretrained', return_value=mock_model) as mock_model_cls:
            async with HuggingFaceEmbedder(hf_config) as embedder:
                assert embedder.config == hf_config
                assert embedder.tokenizer == mock_tokenizer
                assert embedder.model == mock_model
                assert embedder.device == ('cuda' if torch.cuda.is_available() else 'cpu')
                mock_tokenizer_cls.assert_called_once()
                mock_model_cls.assert_called_once()
                mock_model.to.assert_called_once_with(embedder.device)

@pytest.mark.asyncio
async def test_openai_embedder_initialization(openai_config):
    """Test OpenAI embedder initialization."""
    # Test with API key in config
    embedder = OpenAIEmbedder(openai_config)
    assert embedder.config == openai_config
    
    # Test with API key in environment
    config_no_key = EmbeddingConfig(
        name="text-embedding-ada-002",
        type="openai",
        embedding_size=1536,
        max_length=8191
    )
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'env-test-key'}):
        embedder = OpenAIEmbedder(config_no_key)
        assert embedder.config == config_no_key
    
    # Test with no API key
    with pytest.raises(ModelLoadError):
        with patch.dict(os.environ, {}, clear=True):
            OpenAIEmbedder(config_no_key)

@pytest.mark.asyncio
async def test_huggingface_chunking(hf_config):
    """Test HuggingFace embedder chunking functionality."""
    async with HuggingFaceEmbedder(hf_config) as embedder:
        text = "First sentence here. Second one follows. And a third."
        chunks = embedder._chunk_by_sentences(text)
        
        # Verify we get the expected chunks
        assert len(chunks) == 3
        assert chunks[0].strip() == "First sentence here"
        assert chunks[1].strip() == "Second one follows"
        assert chunks[2].strip() == "And a third"

@pytest.mark.asyncio
async def test_late_chunking_spans(hf_config):
    """Test that late chunking embedder returns correct spans."""
    async with LateChunkingEmbedder(hf_config) as embedder:
        # Initialize the chunker
        embedder.chunker = TokenizerBasedSentenceChunker(embedder.tokenizer)
        
        text = "First sentence here. Second one follows."
        chunks, token_spans, char_spans = embedder._chunk_by_sentences(text)
        
        # Verify we get the chunks and spans
        assert len(chunks) == 2
        assert len(token_spans) == len(chunks)
        assert len(char_spans) == len(chunks)
        
        # Verify token spans are valid
        for span in token_spans:
            assert isinstance(span, tuple)
            assert len(span) == 2
            assert span[0] < span[1]
        
        # Verify character spans match the text
        for chunk, span in zip(chunks, char_spans):
            chunk_text = text[span[0]:span[1]].strip()
            assert chunk.strip() == chunk_text.strip()

@pytest.mark.asyncio
async def test_huggingface_embed_chunks(hf_config):
    """Test HuggingFace embedder chunk embedding."""
    # Create mock tokenizer with proper outputs
    mock_tokenizer = Mock()
    
    # Create a proper BatchEncoding object that supports .to()
    class MockBatchEncoding(BatchEncoding):
        def to(self, device):
            # Return self to maintain the same object with device info
            return self
            
    # Mock tokenizer output as BatchEncoding
    mock_tokenizer_output = MockBatchEncoding({
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]),
        "attention_mask": torch.ones((1, 10)),
        "offset_mapping": torch.tensor([[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], 
                                      [5, 6], [6, 7], [7, 8], [8, 9], [9, 10]]])
    })
    
    mock_tokenizer.return_value = mock_tokenizer_output
    mock_tokenizer.convert_tokens_to_ids = Mock(return_value=[4])  # period token
    
    # Create mock model
    mock_model = Mock()
    mock_model.to = Mock(return_value=mock_model)
    mock_output = Mock()
    mock_output.last_hidden_state = torch.randn(1, 10, 768)
    mock_model.return_value = mock_output
    
    with patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
        with patch('transformers.AutoModel.from_pretrained', return_value=mock_model):
            async with HuggingFaceEmbedder(hf_config) as embedder:
                chunks = ["Test chunk 1", "Test chunk 2"]
                embeddings = await embedder.embed_chunks(chunks)
                
                assert isinstance(embeddings, list)
                assert len(embeddings) > 0  # At least one embedding should be produced
                assert all(isinstance(e, ChunkWithEmbedding) for e in embeddings)
                assert all(e.embedding.shape == (hf_config.embedding_size,) for e in embeddings)

@pytest.mark.asyncio
async def test_openai_embed_chunks(openai_config):
    """Test OpenAI embedder chunk embedding."""
    mock_response = {
        'data': [{
            'embedding': np.random.randn(openai_config.embedding_size).tolist()
        }]
    }
    
    mock_acreate = AsyncMock(return_value=mock_response)
    
    with patch('openai.Embedding.acreate', new=mock_acreate):
        embedder = OpenAIEmbedder(openai_config)
        chunks = ["Test chunk"]
        embeddings = await embedder.embed_chunks(chunks)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[1] == openai_config.embedding_size
        mock_acreate.assert_called_once()

@pytest.mark.asyncio
async def test_invalid_chunks():
    """Test handling of invalid chunks."""
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
        embedder = OpenAIEmbedder(EmbeddingConfig(
            name="text-embedding-ada-002",
            type="openai",
            embedding_size=1536,
            max_length=8191
        ))
        with pytest.raises(ValueError):
            await embedder.embed_chunks([])
        with pytest.raises(ValueError):
            await embedder.embed_chunks([1, 2, 3])  # Non-string chunks
