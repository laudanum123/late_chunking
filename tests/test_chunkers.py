"""Tests for text chunking functionality."""
import pytest
from transformers import AutoTokenizer
from late_chunking.chunkers import (
    ChunkMetadata,
    TokenizerBasedSentenceChunker,
    ParagraphChunker,
    FixedTokenChunker
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

@pytest.fixture
def long_text():
    """Sample text long enough to create multiple fixed-size chunks."""
    return " ".join(["This is sentence number " + str(i) + "." for i in range(50)])

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

def test_fixed_token_chunker_basic(tokenizer, sample_text):
    """Test basic fixed token chunking with default settings."""
    chunker = FixedTokenChunker(tokenizer)
    chunks = chunker.chunk_text(sample_text)
    
    # With default size 512, sample text should fit in one chunk
    assert len(chunks) == 1
    assert chunks[0].text == sample_text
    assert chunks[0].token_span is None

def test_fixed_token_chunker_small_size(tokenizer, long_text):
    """Test fixed token chunking with small chunk size."""
    chunk_size = 10
    overlap = 2
    chunker = FixedTokenChunker(tokenizer, chunk_size=chunk_size, overlap=overlap)
    chunks = chunker.chunk_text(long_text, return_tokens=True)
    
    # Verify we get multiple chunks
    assert len(chunks) > 1
    
    # Verify chunk sizes
    for chunk in chunks[:-1]:  # All but last chunk
        token_length = chunk.token_span[1] - chunk.token_span[0]
        assert token_length == chunk_size
        
    # Verify overlap
    for i in range(len(chunks) - 1):
        current_end = chunks[i].token_span[1]
        next_start = chunks[i + 1].token_span[0]
        assert current_end - next_start == overlap

def test_fixed_token_chunker_overlap_validation(tokenizer):
    """Test that overlap is properly validated."""
    # Overlap should be reduced if it's too large
    chunker = FixedTokenChunker(tokenizer, chunk_size=10, overlap=15)
    assert chunker.overlap == 9  # Should be chunk_size - 1
    
    # Normal overlap should be unchanged
    chunker = FixedTokenChunker(tokenizer, chunk_size=10, overlap=5)
    assert chunker.overlap == 5

def test_fixed_token_chunker_empty_text(tokenizer):
    """Test fixed token chunking with empty text."""
    chunker = FixedTokenChunker(tokenizer)
    chunks = chunker.chunk_text("")
    assert len(chunks) == 0
    
    chunks = chunker.chunk_text("   ")
    assert len(chunks) == 0

def test_fixed_token_chunker_character_spans(tokenizer, long_text):
    """Test that character spans are correct."""
    chunker = FixedTokenChunker(tokenizer, chunk_size=20, overlap=5)
    chunks = chunker.chunk_text(long_text)
    
    # Verify character spans
    for chunk in chunks:
        # Spans should be valid
        assert chunk.char_span[0] < chunk.char_span[1]
        # Text should match span
        assert long_text[chunk.char_span[0]:chunk.char_span[1]] == chunk.text
        
    # Verify spans are sequential
    for i in range(len(chunks) - 1):
        # There should be some text overlap due to token overlap
        assert chunks[i].char_span[1] > chunks[i + 1].char_span[0]

def test_fixed_token_chunker_short_text(tokenizer):
    """Test fixed token chunking with text shorter than chunk size."""
    short_text = "Just a very short text."
    chunker = FixedTokenChunker(tokenizer, chunk_size=100)
    chunks = chunker.chunk_text(short_text, return_tokens=True)
    
    # Should get exactly one chunk
    assert len(chunks) == 1
    assert chunks[0].text == short_text
    # Token span should cover all tokens
    token_length = chunks[0].token_span[1] - chunks[0].token_span[0]
    assert token_length < chunker.chunk_size
