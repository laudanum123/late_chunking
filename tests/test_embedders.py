"""Tests for embedder implementations."""
import pytest
import numpy as np
from pathlib import Path
import os
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import torch
import tempfile
import shutil
from transformers import BatchEncoding, PreTrainedTokenizer

from late_chunking.embedders import (
    EmbeddingConfig,
    HuggingFaceEmbedder,
    OpenAIEmbedder,
    ModelLoadError,
    EmbeddingProcessError,
    LateChunkingEmbedder
)
from late_chunking.chunkers import TokenizerBasedSentenceChunker, ChunkMetadata
from late_chunking.embedders.base import ChunkWithEmbedding

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
async def test_huggingface_embedder_initialization(hf_config, temp_dir):
    """Test HuggingFace embedder initialization."""
    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer, \
         patch("transformers.AutoModel.from_pretrained") as mock_model:
            
        # Setup mock tokenizer and model
        mock_tokenizer.return_value = Mock()
        mock_model.return_value = Mock()
        mock_model.return_value.config = Mock(hidden_size=768)
        mock_model.return_value.to = Mock(return_value=mock_model.return_value)
        
        embedder = HuggingFaceEmbedder(hf_config, vector_store_dir=temp_dir)
        assert embedder.model is not None
        assert embedder.tokenizer is not None
        assert embedder.device == ('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.mark.asyncio
async def test_openai_embedder_initialization(openai_config, temp_dir):
    """Test OpenAI embedder initialization."""
    embedder = OpenAIEmbedder(openai_config, vector_store_dir=temp_dir)
    assert embedder.api_key == "test-key"
    assert embedder.model_name == "text-embedding-ada-002"

@pytest.mark.asyncio
async def test_huggingface_chunking(hf_config, temp_dir):
    """Test HuggingFace embedder chunking functionality."""
    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer, \
         patch("transformers.AutoModel.from_pretrained") as mock_model:
            
        # Create properly sized tensors for a test sentence
        # "This is a test sentence."
        # [This] [is] [a] [test] [sentence] [.]
        input_ids = torch.tensor([[2, 3, 1, 5, 6, 4]])  # Maps to: this,is,UNK,test,sentence,.
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1]])
        offset_mapping = torch.tensor([[
            [0, 4],   # This -> id 2
            [5, 7],   # is -> id 3
            [8, 9],   # a -> id 1 (UNK)
            [10, 14], # test -> id 5
            [15, 23], # sentence -> id 6
            [23, 24]  # . -> id 4
        ]])

        # Create a BatchEncoding-like object that supports proper tensor indexing
        class MockBatchEncoding(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.data = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "offset_mapping": offset_mapping
                }
                # Initialize the dict with the same data
                super().__init__(self.data)

            def __getitem__(self, key):
                return self.data[key]

        # Create a proper mock tokenizer class
        class MockTokenizer(PreTrainedTokenizer):
            def __init__(self):
                # Create a basic vocabulary
                self._vocab = {
                    "[PAD]": 0,
                    "[UNK]": 1,
                    "this": 2,
                    "is": 3,
                    ".": 4,
                    "test": 5,
                    "sentence": 6,
                }
                super().__init__(model_max_length=512, padding_side="right", pad_token="[PAD]")
                self.mock_inputs = MockBatchEncoding()

            def get_vocab(self):
                return self._vocab.copy()

            def _tokenize(self, text):
                return ["[MOCK]"]

            def _convert_token_to_id(self, token):
                return self._vocab.get(token.lower(), self._vocab["[UNK]"])

            def convert_tokens_to_ids(self, tokens):
                if isinstance(tokens, str):
                    tokens = [tokens]
                return [self._convert_token_to_id(token) for token in tokens]

            def __call__(self, text, **kwargs):
                return self.mock_inputs

            @property
            def vocab_size(self):
                return len(self._vocab)

            def save_vocabulary(self, save_directory, filename_prefix=None):
                return [], None  # Return empty vocab paths as we don't need to save

        # Create the mock inputs and tokenizer
        mock_tokenizer_instance = MockTokenizer()

        # Set up the tokenizer mock to return our instance
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Set up model mock
        mock_model.return_value = Mock()
        mock_model.return_value.config = Mock(hidden_size=768)
        mock_model.return_value.to = Mock(return_value=mock_model.return_value)
        
        # Create the embedder - the mock tokenizer will be used during initialization
        embedder = HuggingFaceEmbedder(hf_config, vector_store_dir=temp_dir)
        
        # Replace the chunker's tokenizer with our mock instance to ensure consistent mocking
        embedder.chunker.tokenizer = mock_tokenizer_instance
        
        text = "This is a test sentence."
        chunks = await embedder.chunk_text(text)
        assert len(chunks) > 0
        assert all(isinstance(chunk, ChunkMetadata) for chunk in chunks)

@pytest.mark.asyncio
async def test_late_chunking_spans(hf_config, temp_dir):
    """Test that late chunking embedder returns correct spans."""
    embedder = LateChunkingEmbedder(hf_config, vector_store_dir=temp_dir)

    # Mock the embedding function
    async def mock_embed(*args, **kwargs):
        return torch.randn(1, 768)
    embedder._embed_text = mock_embed

    # Use embedder as context manager to initialize chunker
    async with embedder:
        text = "First sentence. Second sentence. Third sentence."
        chunks = await embedder.chunk_text(text)

        # Verify we got three chunks (one per sentence)
        assert len(chunks) == 3
        
        # Verify chunk spans - don't check for period since it's stripped
        assert chunks[0].text == "First sentence"
        assert chunks[1].text == "Second sentence"
        assert chunks[2].text == "Third sentence"

@pytest.mark.asyncio
async def test_huggingface_embed_chunks(hf_config, temp_dir):
    """Test HuggingFace embedder chunk embedding."""
    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer, \
         patch("transformers.AutoModel.from_pretrained") as mock_model:

        # Setup mock tokenizer and model
        mock_tokenizer.return_value = Mock()
        
        # Create tensors with proper shapes
        batch_size = 2
        seq_length = 5
        hidden_size = 768
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5]] * batch_size)  # [2, 5]
        attention_mask = torch.tensor([[1, 1, 1, 1, 1]] * batch_size)  # [2, 5]
        last_hidden_state = torch.randn(batch_size, seq_length, hidden_size)  # [2, 5, 768]

        # Create a proper BatchEncoding-like object that supports dict unpacking
        class MockBatchEncoding(dict):
            def __init__(self, data):
                super().__init__(data)
                self.data = data
            
            def __getitem__(self, key):
                return self.data[key]
            
            def to(self, device):
                return self
        
        # Create mock tensors with all required operations
        class MockTensor:
            @staticmethod
            def __torch_function__(func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                
                # Convert MockTensor arguments to real tensors
                def unwrap(x):
                    if isinstance(x, MockTensor):
                        return x.tensor
                    return x
                
                args = tuple(unwrap(arg) for arg in args)
                kwargs = {k: unwrap(v) for k, v in kwargs.items()}
                
                # Call the actual torch function
                result = func(*args, **kwargs)
                
                # Wrap the result back in MockTensor if it's a tensor
                if isinstance(result, torch.Tensor):
                    return MockTensor(result)
                return result

            def __init__(self, tensor):
                self.tensor = tensor
                self._shape = tensor.shape

            def size(self):
                return self._shape

            def unsqueeze(self, dim):
                return MockTensor(self.tensor.unsqueeze(dim))

            def expand(self, *sizes):
                # Handle both tuple and individual arguments
                if len(sizes) == 1:
                    if isinstance(sizes[0], (tuple, list)):
                        sizes = sizes[0]
                    elif isinstance(sizes[0], Mock):
                        # For attention mask expansion, preserve hidden_size dimension
                        sizes = (-1, -1, 768)
                    else:
                        try:
                            sizes = sizes[0].size()
                        except (AttributeError, TypeError):
                            sizes = (-1, -1, 768)
                result = self.tensor.expand(*sizes)
                print(f"Expand result shape: {result.shape}")  # Debug
                return MockTensor(result)

            def float(self):
                return MockTensor(self.tensor.float())

            def to(self, device):
                return self

            def sum(self, dim):
                # For mean pooling, we want to preserve the embedding dimension (hidden_size)
                # keepdim=True to maintain the dimension for proper broadcasting
                result = self.tensor.sum(dim, keepdim=True)
                print(f"Sum result shape: {result.shape}")  # Debug
                return MockTensor(result)

            def cpu(self):
                return self

            def numpy(self):
                # Ensure we're returning a numpy array with the right shape
                # Squeeze out any extra dimensions except the last (hidden_size)
                arr = self.tensor.detach().cpu()
                # Keep only the last dimension (hidden_size)
                if len(arr.shape) > 1:
                    arr = arr.squeeze(1)  # Remove the middle dimension if present
                if len(arr.shape) > 1:
                    arr = arr[0]  # Take first element if still has batch dimension
                arr = arr.numpy()
                print(f"Numpy conversion shape: {arr.shape}")  # Debug
                return arr

            def __mul__(self, other):
                if isinstance(other, MockTensor):
                    result = self.tensor * other.tensor
                elif isinstance(other, Mock):
                    # For attention mask expansion, preserve hidden_size dimension
                    result = self.tensor.expand(-1, -1, 768)
                else:
                    result = self.tensor * other
                print(f"Multiplication result shape: {result.shape}")  # Debug
                return MockTensor(result)

            def __rmul__(self, other):
                if isinstance(other, Mock):
                    # For attention mask expansion, preserve hidden_size dimension
                    result = self.tensor.expand(-1, -1, 768)
                else:
                    result = other * self.tensor
                print(f"Right multiplication result shape: {result.shape}")  # Debug
                return MockTensor(result)

            def __truediv__(self, other):
                if isinstance(other, MockTensor):
                    result = self.tensor / other.tensor
                elif isinstance(other, Mock):
                    result = self.tensor
                else:
                    # Ensure division maintains the correct shape
                    if isinstance(other, torch.Tensor):
                        result = self.tensor / other
                    else:
                        # For scalar division, just divide the tensor
                        result = self.tensor / other
                print(f"Division result shape: {result.shape}")  # Debug
                return MockTensor(result)

            def squeeze(self, dim=None):
                if dim is None:
                    return MockTensor(self.tensor.squeeze())
                return MockTensor(self.tensor.squeeze(dim))

            def __getattr__(self, name):
                # Forward any other attributes to the underlying tensor
                return getattr(self.tensor, name)

            def detach(self):
                return self

        # Create mock tensors
        mock_attention_mask = MockTensor(attention_mask)  # [2, 5]
        # Initialize last_hidden_state with ones to make shapes more predictable
        mock_last_hidden_state = MockTensor(last_hidden_state)  # [2, 5, 768]

        mock_encoding = MockBatchEncoding({
            "input_ids": input_ids,
            "attention_mask": mock_attention_mask,
        })
        mock_tokenizer.return_value.return_value = mock_encoding

        # Create a model mock that properly handles the forward pass
        class MockOutput:
            def __init__(self, hidden_states):
                # Ensure last_hidden_state is a MockTensor
                if isinstance(hidden_states, MockTensor):
                    self.last_hidden_state = hidden_states
                else:
                    self.last_hidden_state = MockTensor(hidden_states)

        model_mock = Mock()
        model_mock.config = Mock(hidden_size=hidden_size)
        model_mock.to = Mock(return_value=model_mock)

        # Setup the forward pass to return proper tensor
        def mock_forward(**kwargs):
            return MockOutput(mock_last_hidden_state)
        model_mock.__call__ = mock_forward
        
        mock_model.return_value = model_mock

        embedder = HuggingFaceEmbedder(hf_config, vector_store_dir=temp_dir)
        
        # Create test chunks
        chunks = [
            ChunkMetadata(text="Test chunk 1", char_span=(0, 11)),
            ChunkMetadata(text="Test chunk 2", char_span=(12, 23))
        ]

        # Test embedding
        embedded_chunks = await embedder.embed_chunks(chunks)
        
        # Debug output
        print(f"\nEmbedding shapes:")
        for i, chunk in enumerate(embedded_chunks):
            print(f"Chunk {i} embedding shape: {chunk.embedding.shape}")

        # Verify embeddings
        assert len(embedded_chunks) == batch_size
        assert all(isinstance(chunk, ChunkWithEmbedding) for chunk in embedded_chunks)
        assert all(chunk.embedding.shape == (hidden_size,) for chunk in embedded_chunks)

@pytest.mark.asyncio
async def test_openai_embed_chunks(openai_config, temp_dir):
    """Test OpenAI embedder chunk embedding."""
    with patch("openai.Embedding.create") as mock_embed:
        mock_embed.return_value = {"data": [{"embedding": [0.1] * 1536}]}
        
        embedder = OpenAIEmbedder(openai_config, vector_store_dir=temp_dir)
        chunks = [ChunkMetadata(text="Test chunk", char_span=(0, 10))]
        
        embedded_chunks = await embedder.embed_chunks(chunks)
        assert len(embedded_chunks) == 1
        assert embedded_chunks[0].embedding.shape == (1536,)

@pytest.mark.asyncio
async def test_invalid_chunks(temp_dir):
    """Test handling of invalid chunks."""
    config = EmbeddingConfig(
        name="test-model",
        type="huggingface",
        embedding_size=768,
        max_length=512
    )

    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer, \
         patch("transformers.AutoModel.from_pretrained") as mock_model:

        # Setup mock tokenizer and model
        mock_tokenizer.return_value = Mock()
        mock_model.return_value = Mock()
        mock_model.return_value.config = Mock(hidden_size=768)
        mock_model.return_value.to = Mock(return_value=mock_model.return_value)

        embedder = HuggingFaceEmbedder(config, vector_store_dir=temp_dir)
        
        # Test with invalid chunks (strings instead of ChunkMetadata)
        with pytest.raises(EmbeddingProcessError) as exc_info:
            await embedder.embed_chunks(["not a chunk metadata"])
        
        # Verify the error message
        assert "Failed to embed chunks: 'str' object has no attribute 'text'" in str(exc_info.value)
