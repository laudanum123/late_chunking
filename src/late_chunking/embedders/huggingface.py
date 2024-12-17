"""HuggingFace-based embedder implementation."""
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Optional
from pathlib import Path
import logging
import faiss
import pickle

from .base import (
    BaseEmbedder,
    EmbeddingConfig,
    ModelLoadError,
    EmbeddingProcessError,
    ChunkWithEmbedding
)
from ..chunkers import TokenizerBasedSentenceChunker, ChunkMetadata

logger = logging.getLogger(__name__)

class HuggingFaceEmbedder(BaseEmbedder):
    """Embedder implementation using HuggingFace models."""
    
    def __init__(self, config: EmbeddingConfig, vector_store_dir: Optional[str] = None):
        """Initialize HuggingFace embedder.
        
        Args:
            config: EmbeddingConfig instance
            vector_store_dir: Optional path to vector store directory
        """
        super().__init__(config, vector_store_dir)
        self.tokenizer = None
        self.model = None
        self.device = None
        self.chunks = []
        self.index = None
        self.batch_size = config.additional_params.get('batch_size', 32) if config.additional_params else 32
        
        # Initialize model and chunker
        self._initialize_model()
        self.chunker = TokenizerBasedSentenceChunker(self.tokenizer)
        
    def _initialize_model(self):
        """Initialize the model and tokenizer."""
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
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {str(e)}") from e

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if hasattr(self, 'model'):
            self.model.cpu()
            torch.cuda.empty_cache()
            logger.info("Model resources cleaned up")

    async def chunk_text(self, text: str) -> List[ChunkMetadata]:
        """Chunk text into sentences.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunk metadata
        """
        return self.chunker.chunk_text(text)

    def _chunk_by_sentences(self, chunk: ChunkMetadata) -> List[str]:
        """Split text into sentences.

        Args:
            chunk: Chunk metadata to split into sentences

        Returns:
            List of sentence chunks
        """
        return [chunk.text] if isinstance(chunk, ChunkMetadata) else self.chunker.chunk_text(chunk, return_tokens=False)

    async def embed_chunks(self, chunks: List[ChunkMetadata]) -> List[ChunkWithEmbedding]:
        """Embed a list of text chunks.
        
        Args:
            chunks: List of text chunks to embed
            
        Returns:
            List of chunk embeddings
            
        Raises:
            EmbeddingProcessError: If embedding fails
        """
        try:
            all_embeddings = []
            embeddings_array = []
            
            # Process chunks in batches
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i:i + self.batch_size]
                
                # Tokenize all texts in batch
                inputs = self.tokenizer(
                    [chunk.text for chunk in batch],
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                # Get embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use mean pooling
                    attention_mask = inputs["attention_mask"]
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    embeddings_sum = torch.sum(token_embeddings * input_mask_expanded, 1)
                    mask_sum = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    embeddings = (embeddings_sum / mask_sum).cpu().numpy()
                    
                # Normalize embeddings
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings_array.extend(embeddings)
                
                # Create ChunkWithEmbedding instances
                for chunk, embedding in zip(batch, embeddings):
                    all_embeddings.append(ChunkWithEmbedding(
                        text=chunk.text,
                        embedding=embedding,
                        char_span=chunk.char_span,
                        token_span=chunk.token_span
                    ))
            
            # Add embeddings to vector store
            self._add_embeddings(np.array(embeddings_array), all_embeddings)
            
            return all_embeddings
            
        except Exception as e:
            raise EmbeddingProcessError(f"Failed to embed chunks: {str(e)}") from e

    async def embed_query(self, query: str) -> np.ndarray:
        """Embed a query string without adding to vector store.
        
        Args:
            query: Query string to embed
            
        Returns:
            Query embedding as numpy array
        """
        try:
            # Tokenize and get embeddings
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
