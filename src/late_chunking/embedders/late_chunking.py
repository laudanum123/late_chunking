"""Late chunking embedder implementation."""
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Tuple, Optional, Type
import logging
from pathlib import Path
import faiss
import pickle

from .base import (
    BaseEmbedder,
    EmbeddingConfig,
    ModelLoadError,
    EmbeddingProcessError,
    ChunkWithEmbedding
)
from ..chunkers import (
    Chunker,
    TokenizerBasedSentenceChunker,
    FixedTokenChunker,
    ParagraphChunker,
    ChunkMetadata
)

logger = logging.getLogger(__name__)

class LateChunkingEmbedder(BaseEmbedder):
    """Embedder using late chunking strategy."""
    
    def __init__(
        self,
        config: EmbeddingConfig,
        vector_store_dir: Optional[str] = None,
        chunker_class: Type[Chunker] = TokenizerBasedSentenceChunker,
        chunker_params: Optional[dict] = None
    ):
        """Initialize late chunking embedder.
        
        Args:
            config: EmbeddingConfig instance
            vector_store_dir: Optional path to vector store directory
            chunker_class: Chunking strategy class to use (default: TokenizerBasedSentenceChunker)
            chunker_params: Optional parameters for the chunker
        """
        super().__init__(config, vector_store_dir)
        self.tokenizer = None
        self.model = None
        self.device = None
        self.chunks = []
        self.index = None
        self.batch_size = config.additional_params.get('batch_size', 32) if config.additional_params else 32
        self.vector_store_path = None
        self.chunker_class = chunker_class
        self.chunker_params = chunker_params or {}
        self.chunker = None

    def set_vector_store_path(self, path: Path):
        """Set the path for vector store operations."""
        self.vector_store_path = path

    async def __aenter__(self):
        """Async context manager entry."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.name,
                trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                self.config.name,
                trust_remote_code=True
            )
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(self.device)
            
            # Initialize chunker with tokenizer and any additional params
            if self.chunker_class == TokenizerBasedSentenceChunker:
                self.chunker = self.chunker_class(self.tokenizer)
            elif self.chunker_class == FixedTokenChunker:
                self.chunker = self.chunker_class(
                    tokenizer=self.tokenizer,
                    chunk_size=self.chunker_params.get('chunk_size', 512),
                    overlap=self.chunker_params.get('overlap', 50)
                )
            else:
                self.chunker = self.chunker_class(**self.chunker_params)
            
            logger.info(f"Late chunking embedder initialized on {self.device}")
            return self
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {str(e)}") from e

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if hasattr(self, 'model'):
            self.model.cpu()
            torch.cuda.empty_cache()
            logger.info("Late chunking embedder resources cleaned up")

    def _chunk_by_sentences(self, input_text: str) -> Tuple[List[str], List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Split text into sentences and return chunks with spans."""
        chunks = self.chunker.chunk_text(input_text, return_tokens=True)
        
        # Extract the components
        texts = [chunk.text for chunk in chunks]
        token_spans = [chunk.token_span for chunk in chunks]
        char_spans = [chunk.char_span for chunk in chunks]
        
        return texts, token_spans, char_spans

    def _late_chunking(self, token_embeddings: torch.Tensor, span_annotations: List[Tuple[int, int]]) -> List[np.ndarray]:
        """Create embeddings using late chunking approach.
        
        Args:
            token_embeddings: Tensor of shape (batch_size, seq_len, hidden_size)
            span_annotations: List of (start, end) token spans
            
        Returns:
            List of numpy arrays containing chunk embeddings
        """
        # Handle batched input - we only support batch_size=1 for now
        embeddings = token_embeddings[0]  # Shape: (seq_len, hidden_size)
        
        # Filter out None spans
        span_annotations = [span for span in span_annotations if span is not None]
        
        # Truncate spans if needed
        if self.config.max_length is not None:
            span_annotations = [
                (start, min(end, self.config.max_length))
                for (start, end) in span_annotations
                if start < self.config.max_length
            ]
        
        # Pool embeddings for each span
        pooled_embeddings = []
        for start, end in span_annotations:
            if end <= start:
                continue
                
            # Mean pooling over the span
            span_embeddings = embeddings[start:end]
            if len(span_embeddings) == 0:
                continue
                
            pooled = span_embeddings.mean(dim=0)
            
            # L2 normalize, handling zero vectors
            norm = torch.norm(pooled)
            if norm > 0:
                pooled = pooled / norm
            else:
                # If norm is 0, use a random unit vector
                pooled = torch.randn_like(pooled)
                pooled = pooled / torch.norm(pooled)
                
            pooled_embeddings.append(pooled.detach().cpu().numpy())
        
        return pooled_embeddings

    def _create_macro_chunks(self, text: str) -> List[Tuple[str, int]]:
        """Split text into macro chunks that fit within model's context size.
        
        Args:
            text: Input text to split
            
        Returns:
            List of (chunk_text, token_offset) tuples
        """
        if not self.config.max_length:
            return [(text, 0)]
            
        # Tokenize the full text to get token counts
        inputs = self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt"
        )
        token_ids = inputs["input_ids"][0]
        
        # If text fits in context window, return as single chunk
        if len(token_ids) <= self.config.max_length:
            return [(text, 0)]
            
        # Split into chunks that fit within context window
        chunks = []
        start_idx = 0
        token_offset = 0
        
        while start_idx < len(token_ids):
            # Get chunk size that fits in context window
            end_idx = min(start_idx + self.config.max_length, len(token_ids))
            
            # Decode tokens back to text
            chunk_tokens = token_ids[start_idx:end_idx]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            chunks.append((chunk_text, token_offset))
            token_offset += len(chunk_tokens)
            start_idx = end_idx
            
        return chunks

    async def chunk_text(self, text: str) -> List[ChunkMetadata]:
        """Chunk text using the configured chunking strategy.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunk metadata
        """
        if self.chunker is None:
            # Initialize chunker with tokenizer and any additional params
            if self.chunker_class == TokenizerBasedSentenceChunker:
                self.chunker = self.chunker_class(self.tokenizer)
            elif self.chunker_class == FixedTokenChunker:
                self.chunker = self.chunker_class(
                    tokenizer=self.tokenizer,
                    chunk_size=self.chunker_params.get('chunk_size', 512),
                    overlap=self.chunker_params.get('overlap', 50)
                )
            else:
                self.chunker = self.chunker_class(**self.chunker_params)
            
        # First create macro chunks to handle long texts
        macro_chunks = self._create_macro_chunks(text)
        
        # Process each macro chunk
        all_chunks = []
        for chunk_text, offset in macro_chunks:
            # Get chunks with token spans for this macro chunk
            chunks = self.chunker.chunk_text(chunk_text, return_tokens=True)
            
            # Adjust token spans for offset
            for chunk in chunks:
                if chunk.token_span is not None:
                    start, end = chunk.token_span
                    chunk.token_span = (start + offset, end + offset)
                    # Store the full text for late chunking
                    chunk.full_text = text
            
            all_chunks.extend(chunks)
            
        return all_chunks

    async def embed_chunks(self, chunks: List[ChunkMetadata]) -> List[ChunkWithEmbedding]:
        """Embed chunks using late chunking strategy.
        
        This method first embeds the full context for each token, then pools
        the token embeddings according to the chunk boundaries.
        
        Args:
            chunks: List of chunks to embed
            
        Returns:
            List of chunks with embeddings
        """
        try:
            # Group chunks by their full text to process together
            text_to_chunks = {}
            for chunk in chunks:
                text_to_chunks.setdefault(chunk.full_text, []).append(chunk)
            
            all_embeddings = []
            all_processed_chunks = []
            
            # Process each unique text
            for text, text_chunks in text_to_chunks.items():
                # Create macro chunks for long texts
                macro_chunks = self._create_macro_chunks(text)
                
                # Process each macro chunk
                for chunk_text, token_offset in macro_chunks:
                    # Get relevant chunks for this macro chunk
                    relevant_chunks = []
                    for chunk in text_chunks:
                        start, end = chunk.token_span
                        # Check if chunk overlaps with macro chunk
                        if start >= token_offset and start < token_offset + self.config.max_length:
                            relevant_chunks.append(chunk)
                    
                    if not relevant_chunks:
                        continue
                        
                    # Get span annotations
                    span_annotations = []
                    for chunk in relevant_chunks:
                        start, end = chunk.token_span
                        # Adjust spans relative to macro chunk
                        span_annotations.append((
                            start - token_offset,
                            end - token_offset
                        ))
                    
                    # Tokenize and get embeddings
                    inputs = self.tokenizer(
                        chunk_text,
                        padding=True,
                        truncation=True,
                        max_length=self.config.max_length,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    
                    # Get embeddings for each span
                    chunk_embeddings = self._late_chunking(
                        outputs.last_hidden_state,
                        span_annotations
                    )
                    
                    all_embeddings.extend(chunk_embeddings)
                    all_processed_chunks.extend(relevant_chunks)

            # Create ChunkWithEmbedding objects for chunks with valid embeddings
            embedded_chunks = []
            embeddings_array = []
            for chunk, embedding in zip(all_processed_chunks, all_embeddings):
                if embedding is not None:
                    chunk_with_embedding = ChunkWithEmbedding(
                        text=chunk.text,
                        embedding=embedding,
                        char_span=chunk.char_span,
                        token_span=chunk.token_span
                    )
                    embedded_chunks.append(chunk_with_embedding)
                    embeddings_array.append(embedding)
            
            # Add embeddings to vector store
            if embeddings_array:
                self._add_embeddings(np.array(embeddings_array), embedded_chunks)
            
            return embedded_chunks

        except Exception as e:
            raise EmbeddingProcessError(f"Failed to embed chunks: {str(e)}") from e

    async def embed_query(self, query: str) -> np.ndarray:
        """Embed a query string without adding to vector store.
        
        For queries, we use the traditional embedding approach since we don't need
        late chunking for single short queries.
        
        Args:
            query: Query string to embed
            
        Returns:
            Query embedding as numpy array
        """
        try:
            # For queries, use traditional embedding approach
            inputs = self.tokenizer(
                query,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Use mean pooling to get embeddings
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings_sum = torch.sum(token_embeddings * input_mask_expanded, 1)
            mask_sum = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = (embeddings_sum / mask_sum).cpu().numpy()[0]
            
            # Normalize the embedding
            return embedding / np.linalg.norm(embedding)
            
        except Exception as e:
            logger.error(f"Error computing query embedding: {str(e)}")
            raise EmbeddingProcessError(f"Failed to compute query embedding: {str(e)}") from e

    def save_vector_store(self):
        """Save the current vector store state to disk."""
        try:
            if self.vector_store_path and self.index is not None:
                self.vector_store_path.mkdir(parents=True, exist_ok=True)
                index_path = self.vector_store_path / "index.faiss"
                chunks_path = self.vector_store_path / "chunks.pkl"
                
                # Count embeddings by document ID
                doc_id_counts = {}
                for chunk in self.chunks:
                    doc_id = getattr(chunk, 'doc_id', 'unknown')
                    doc_id_counts[doc_id] = doc_id_counts.get(doc_id, 0) + 1
                
                # Log counts
                logger.info(f"Saving vector store to {self.vector_store_path}")
                for doc_id, count in doc_id_counts.items():
                    logger.info(f"Document {doc_id}: {count} embeddings")
                
                faiss.write_index(self.index, str(index_path))
                with open(chunks_path, 'wb') as f:
                    pickle.dump(self.chunks, f)
                logger.info(f"Total embeddings saved: {len(self.chunks)}")
        except Exception as e:
            logger.warning(f"Error saving vector store: {str(e)}")

    def _load_vector_store(self):
        """Load vector store from disk if it exists."""
        try:
            if self.vector_store_path:
                index_path = self.vector_store_path / "index.faiss"
                chunks_path = self.vector_store_path / "chunks.pkl"
                
                if index_path.exists() and chunks_path.exists():
                    self.index = faiss.read_index(str(index_path))
                    with open(chunks_path, 'rb') as f:
                        self.chunks = pickle.load(f)
                    return True
        except Exception as e:
            logger.warning(f"Error loading vector store: {str(e)}")
        return False
    
    def _add_embeddings(self, embeddings_array, chunks):
        """Add new embeddings to vector store."""
        try:
            if self.index is None:
                self.index = faiss.IndexFlatL2(embeddings_array.shape[1])
            
            self.index.add(embeddings_array)
            self.chunks.extend(chunks)
                
        except Exception as e:
            logger.warning(f"Error adding to vector store: {str(e)}")
