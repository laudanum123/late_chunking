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
        for i in range(len(token_ids)):
            if token_ids[i] == sep_id:
                # Check if this is a real sentence boundary
                if i + 1 < len(token_ids) and (
                    token_offsets[i + 1][0] - token_offsets[i][1] > 0  # Gap between tokens
                    or token_ids[i + 1] == sep_id  # Multiple periods
                ):
                    # Get position before the period
                    chunk_positions.append((i, token_offsets[i][0].item()))
        
        # Add the last position to capture the final sentence
        if len(token_ids) > 0:
            last_token_idx = len(token_ids) - 1
            # If the last token is a period, use its start position
            if token_ids[last_token_idx] == sep_id:
                last_pos = (last_token_idx, token_offsets[last_token_idx][0].item())
            else:
                last_pos = (last_token_idx, token_offsets[last_token_idx][1].item())
                
            if not chunk_positions or last_pos[1] > chunk_positions[-1][1]:
                chunk_positions.append(last_pos)
            
        # Create chunks with spans
        chunks = []
        prev_token_idx = 0
        prev_char_pos = 0
        
        for token_idx, char_pos in chunk_positions:
            chunk_text = text[prev_char_pos:char_pos].strip()
            if chunk_text:  # Only add non-empty chunks
                metadata = ChunkMetadata(
                    text=chunk_text,
                    char_span=(prev_char_pos, char_pos),
                    token_span=(prev_token_idx, token_idx) if return_tokens else None
                )
                chunks.append(metadata)
            prev_token_idx = token_idx + 1
            # Skip over the period and any whitespace
            if token_idx + 1 < len(token_offsets):
                prev_char_pos = token_offsets[token_idx + 1][0].item()
            else:
                prev_char_pos = char_pos
            
        # Add the final chunk if there's remaining text
        if prev_char_pos < len(text):
            final_text = text[prev_char_pos:].strip()
            if final_text:
                # Remove trailing period if present
                if final_text.endswith('.'):
                    final_text = final_text[:-1].strip()
                if final_text:  # Check again in case it was just a period
                    metadata = ChunkMetadata(
                        text=final_text,
                        char_span=(prev_char_pos, len(text)),
                        token_span=(prev_token_idx, len(token_ids)) if return_tokens else None
                    )
                    chunks.append(metadata)
        
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
                    char_span=(current_pos, para_end)
                )
                chunks.append(metadata)
                current_pos = para_end
                
        return chunks
