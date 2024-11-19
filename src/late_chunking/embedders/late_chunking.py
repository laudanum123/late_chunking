"""Late chunking embedder implementation."""
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Tuple, Optional, Type
import logging
from pathlib import Path
import faiss
import pickle

from .base import BaseEmbedder, EmbeddingConfig, ModelLoadError, EmbeddingProcessError
from .vector_store import ChunkWithEmbedding
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
            else:
                self.chunker = self.chunker_class(
                    tokenizer=self.tokenizer,
                    **self.chunker_params
                )
            
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

    def _late_chunking(self, model_output: torch.Tensor, span_annotations: List[Tuple[int, int]]) -> List[np.ndarray]:
        """Create embeddings using late chunking approach."""
        token_embeddings = model_output[0]
        outputs = []
        
        for embeddings, annotations in zip(token_embeddings, [span_annotations]):
            if self.config.max_length is not None:
                annotations = [
                    (start, min(end, self.config.max_length - 1))
                    for (start, end) in annotations
                    if start < (self.config.max_length - 1)
                ]
            
            pooled_embeddings = [
                embeddings[start:end].sum(dim=0) / (end - start)
                for start, end in annotations
                if (end - start) >= 1
            ]
            # L2 normalize the embeddings
            pooled_embeddings = [
                (embedding / torch.norm(embedding)).detach().cpu().numpy()
                for embedding in pooled_embeddings
            ]
            outputs.append(pooled_embeddings)
        
        return outputs[0]

    def _create_macro_chunks(self, text: str) -> List[Tuple[str, int]]:
        """Split text into macro chunks that fit within model's context size."""
        if not self.config.max_length:
            return [(text, 0)]
            
        # Get rough character limit based on max tokens
        # Using a conservative estimate of 4 characters per token
        char_limit = self.config.max_length * 4
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Find a good breaking point near the character limit
            end = min(start + char_limit, len(text))
            
            # If we're not at the end, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings (.!?) followed by whitespace
                last_period = max(
                    text.rfind(". ", start, end),
                    text.rfind("! ", start, end),
                    text.rfind("? ", start, end)
                )
                
                if last_period != -1:
                    end = last_period + 1
            
            chunks.append((text[start:end], start))
            start = end
            
        return chunks

    async def chunk_text(self, text: str) -> List[ChunkMetadata]:
        """Chunk text using the configured chunking strategy.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunk metadata
        """
        if not self.chunker:
            raise ValueError("Chunker not initialized. Use embedder as context manager.")
        return self.chunker.chunk_text(text)

    async def embed_chunks(self, chunks: List[ChunkMetadata]) -> List[ChunkWithEmbedding]:
        """Embed chunks using late chunking strategy.
        
        This method first embeds the full context for each token, then pools
        the token embeddings according to the chunk boundaries.
        
        Args:
            chunks: List of chunks to embed
            
        Returns:
            List of chunks with embeddings
        """
        if not chunks:
            return []

        try:
            # First, get the full text context for each chunk
            all_embeddings = []
            for chunk in chunks:
                # Tokenize the full text context
                inputs = self.tokenizer(
                    chunk.text,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt"
                ).to(self.device)

                # Get embeddings with full context
                with torch.no_grad():
                    outputs = self.model(**inputs)

                # Use mean pooling over the chunk's token span
                attention_mask = inputs["attention_mask"]
                token_embeddings = outputs.last_hidden_state
                
                if chunk.token_span:
                    start, end = chunk.token_span
                    # Only pool over the chunk's tokens
                    chunk_embeddings = token_embeddings[:, start:end, :]
                    chunk_mask = attention_mask[:, start:end]
                else:
                    # If no token span, use the whole sequence
                    chunk_embeddings = token_embeddings
                    chunk_mask = attention_mask

                # Mean pooling
                mask_expanded = chunk_mask.unsqueeze(-1).expand(chunk_embeddings.size()).float()
                sum_embeddings = torch.sum(chunk_embeddings * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                embedding = (sum_embeddings / sum_mask).cpu().numpy()[0]

                # Normalize embedding
                embedding = embedding / np.linalg.norm(embedding)
                all_embeddings.append(embedding)

            # Create ChunkWithEmbedding objects
            return [
                ChunkWithEmbedding(
                    text=chunk.text,
                    embedding=embedding,
                    char_span=chunk.char_span,
                    token_span=chunk.token_span
                )
                for chunk, embedding in zip(chunks, all_embeddings)
            ]

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
