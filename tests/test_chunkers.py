"""Tests for text chunking functionality."""
import pytest
from transformers import AutoTokenizer
from late_chunking.chunkers import (
    ChunkMetadata,
    TokenizerBasedSentenceChunker,
    ParagraphChunker
)

@pytest.fixture
def tokenizer():
    """Fixture for HuggingFace tokenizer."""
    return AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v2-base-en")

@pytest.fixture
def sample_text():
    """Sample text for testing chunking."""
    return "This is a test sentence. Another sentence follows. And here's the last one."

@pytest.fixture
def sample_paragraphs():
    """Sample text with paragraphs for testing."""
    return """First paragraph with multiple sentences. Second sentence here.

Second paragraph starts here. It has two sentences.

Final paragraph is short."""

def test_sentence_chunker_basic(tokenizer, sample_text):
    """Test basic sentence chunking without token spans."""
    chunker = TokenizerBasedSentenceChunker(tokenizer)
    chunks = chunker.chunk_text(sample_text, return_tokens=False)
    
    # Verify we get the right number of chunks
    assert len(chunks) == 3
    
    # Verify chunk texts
    expected_texts = [
        "This is a test sentence",
        "Another sentence follows",
        "And here's the last one"
    ]
    assert [chunk.text.strip() for chunk in chunks] == expected_texts
    
    # Verify character spans
    for chunk in chunks:
        assert isinstance(chunk, ChunkMetadata)
        assert chunk.char_span[0] < chunk.char_span[1]
        assert sample_text[chunk.char_span[0]:chunk.char_span[1]].strip() == chunk.text
        assert chunk.token_span is None  # Token spans not requested

def test_sentence_chunker_with_tokens(tokenizer, sample_text):
    """Test sentence chunking with token spans."""
    chunker = TokenizerBasedSentenceChunker(tokenizer)
    chunks = chunker.chunk_text(sample_text, return_tokens=True)
    
    # Verify token spans are included
    for chunk in chunks:
        assert isinstance(chunk.token_span, tuple)
        assert len(chunk.token_span) == 2
        assert chunk.token_span[0] < chunk.token_span[1]

def test_paragraph_chunker(sample_paragraphs):
    """Test paragraph-based chunking."""
    chunker = ParagraphChunker()
    chunks = chunker.chunk_text(sample_paragraphs)
    
    # Verify we get three paragraphs
    assert len(chunks) == 3
    
    # Verify paragraph content
    assert "First paragraph" in chunks[0].text
    assert "Second paragraph" in chunks[1].text
    assert "Final paragraph" in chunks[2].text
    
    # Verify character spans
    for chunk in chunks:
        text_from_span = sample_paragraphs[chunk.char_span[0]:chunk.char_span[1]]
        assert chunk.text in text_from_span

def test_paragraph_chunker_token_spans():
    """Test that paragraph chunker raises error for token spans."""
    chunker = ParagraphChunker()
    with pytest.raises(NotImplementedError):
        chunker.chunk_text("Some text", return_tokens=True)

def test_empty_text(tokenizer):
    """Test chunking with empty text."""
    chunker = ParagraphChunker()
    chunks = chunker.chunk_text("")
    assert len(chunks) == 0
    
    sentence_chunker = TokenizerBasedSentenceChunker(tokenizer)
    chunks = sentence_chunker.chunk_text("")
    assert len(chunks) == 0
