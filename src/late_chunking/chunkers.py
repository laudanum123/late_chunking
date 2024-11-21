"""Text chunking strategies for document splitting."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional
from transformers import PreTrainedTokenizer

@dataclass
class ChunkMetadata:
    """Metadata for a text chunk."""
    text: str
    char_span: Tuple[int, int]  # Character offsets in original text
    token_span: Optional[Tuple[int, int]] = None  # Token indices in the tokenized text
    full_text: Optional[str] = None  # Full text from which this chunk was extracted

class Chunker(ABC):
    """Base class for text chunking strategies."""
    
    @abstractmethod
    def chunk_text(self, text: str, return_tokens: bool = False) -> List[ChunkMetadata]:
        """Split text into chunks.
        
        Args:
            text: Input text to split
            return_tokens: If True, include token spans in the output
            
        Returns:
            List of ChunkMetadata objects
        """
        pass

class TokenizerBasedSentenceChunker(Chunker):
    """Sentence chunker that uses a tokenizer for accurate token spans."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        """Initialize the chunker.
        
        Args:
            tokenizer: HuggingFace tokenizer to use for tokenization
        """
        self.tokenizer = tokenizer
        
    def chunk_text(self, text: str, return_tokens: bool = False) -> List[ChunkMetadata]:
        """Split text into sentences using tokenizer information.
        
        Args:
            text: Text to split into sentences
            return_tokens: If True, include token spans in the output
            
        Returns:
            List of ChunkMetadata objects containing sentences and their spans
        """
        # Handle empty text
        if not text.strip():
            return []
            
        # Tokenize the text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False
        )
        
        token_ids = inputs["input_ids"][0]
        token_offsets = inputs["offset_mapping"][0]
        
        # Handle case where tokenization produced no tokens
        if len(token_ids) == 0:
            return []
        
        # Get the separator token id (usually period)
        sep_id = self.tokenizer.convert_tokens_to_ids(["."])[0]
        
        # Find chunk positions based on sentence boundaries
        chunk_positions = []
        current_sentence_start = 0
        
        for i in range(len(token_ids)):
            if token_ids[i] == sep_id:
                # Check if this is a real sentence boundary
                if i + 1 < len(token_ids) and (
                    token_offsets[i + 1][0] - token_offsets[i][1] > 0  # Gap between tokens
                    or token_ids[i + 1] == sep_id  # Multiple periods
                ):
                    # Get the position before the period for text extraction
                    chunk_positions.append((
                        current_sentence_start,
                        i + 1,  # Include period in token span
                        token_offsets[current_sentence_start][0].item(),
                        token_offsets[i][0].item()  # End before the period for text
                    ))
                    current_sentence_start = i + 1
        
        # Add the last sentence if it doesn't end with a period
        if current_sentence_start < len(token_ids):
            last_token_idx = len(token_ids) - 1
            end_char = token_offsets[last_token_idx][1].item()
            # If last token is a period, exclude it from text
            if token_ids[last_token_idx] == sep_id:
                end_char = token_offsets[last_token_idx][0].item()
            
            chunk_positions.append((
                current_sentence_start,
                last_token_idx + 1,
                token_offsets[current_sentence_start][0].item(),
                end_char
            ))
            
        # Create chunks with spans
        chunks = []
        
        for start_token, end_token, start_char, end_char in chunk_positions:
            # Extract the text for this chunk
            chunk_text = text[start_char:end_char].strip()
            
            # Only create chunk if there's actual text
            if chunk_text:
                chunk = ChunkMetadata(
                    text=chunk_text,
                    char_span=(start_char, end_char),
                    token_span=(start_token, end_token) if return_tokens else None,
                    full_text=text
                )
                chunks.append(chunk)
                
        return chunks

class FixedTokenChunker(Chunker):
    """Split text into chunks of fixed token size with overlap."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, chunk_size: int = 512, overlap: int = 50):
        """Initialize the chunker.
        
        Args:
            tokenizer: HuggingFace tokenizer to use for tokenization
            chunk_size: Number of tokens per chunk (default: 512)
            overlap: Number of tokens to overlap between chunks (default: 50)
        """
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap = min(overlap, chunk_size - 1)  # Ensure overlap is smaller than chunk size
        
    def chunk_text(self, text: str, return_tokens: bool = False) -> List[ChunkMetadata]:
        """Split text into fixed-size chunks with overlap.
        
        Args:
            text: Text to split into chunks
            return_tokens: If True, include token spans in the output
            
        Returns:
            List of ChunkMetadata objects containing chunks and their spans
        """
        # Handle empty text
        if not text.strip():
            return []
            
        # Tokenize the text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False
        )
        
        token_ids = inputs["input_ids"][0]
        token_offsets = inputs["offset_mapping"][0]
        
        # Handle case where tokenization produced no tokens
        if len(token_ids) == 0:
            return []
            
        chunks = []
        start_idx = 0
        
        while start_idx < len(token_ids):
            # Calculate end index for current chunk
            end_idx = min(start_idx + self.chunk_size, len(token_ids))
            
            # Get character spans
            char_start = token_offsets[start_idx][0].item()
            char_end = token_offsets[end_idx - 1][1].item()
            
            # Create chunk metadata
            chunk = ChunkMetadata(
                text=text[char_start:char_end],
                char_span=(char_start, char_end),
                token_span=(start_idx, end_idx) if return_tokens else None,
                full_text=text
            )
            chunks.append(chunk)
            
            # Move start index for next chunk, accounting for overlap
            start_idx = end_idx - self.overlap
            
            # Break if we've reached the end
            if end_idx == len(token_ids):
                break
        
        return chunks

class ParagraphChunker(Chunker):
    """Split text into paragraphs based on double newlines."""
    
    def chunk_text(self, text: str, return_tokens: bool = False) -> List[ChunkMetadata]:
        """Split text into paragraphs.
        
        Args:
            text: Text to split into paragraphs
            return_tokens: If True, include token spans in the output (not supported)
            
        Returns:
            List of ChunkMetadata objects containing paragraphs and their spans
        """
        if return_tokens:
            raise NotImplementedError("Token spans are not supported for ParagraphChunker")
            
        chunks = []
        current_pos = 0
        
        # Split on double newlines
        paragraphs = text.split("\n\n")
        
        for para in paragraphs:
            # Find the actual start position after whitespace
            while current_pos < len(text) and text[current_pos].isspace():
                current_pos += 1
                
            para_text = para.strip()
            if para_text:  # Only add non-empty paragraphs
                # Find the end position in original text
                para_end = current_pos + len(para)
                metadata = ChunkMetadata(
                    text=para_text,
                    char_span=(current_pos, para_end),
                    full_text=text
                )
                chunks.append(metadata)
                current_pos = para_end
                
        return chunks
