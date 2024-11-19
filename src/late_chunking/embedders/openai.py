"""OpenAI-based embedder implementation."""
import openai
import numpy as np
from typing import List, Optional
import os
import asyncio
import logging
from pathlib import Path

from .base import (
    BaseEmbedder,
    EmbeddingConfig,
    ModelLoadError,
    EmbeddingProcessError,
    ChunkWithEmbedding
)
from ..chunkers import ChunkMetadata

logger = logging.getLogger(__name__)

class OpenAIEmbedder(BaseEmbedder):
    """Embedder implementation using OpenAI API."""
    
    def __init__(self, config: EmbeddingConfig, vector_store_dir: Optional[str] = None):
        """Initialize OpenAI embedder.
        
        Args:
            config: Configuration for the embedder
            vector_store_dir: Optional path to vector store directory
        """
        super().__init__(config, vector_store_dir)
        self.api_key = config.api_key
        self.model_name = config.name
        openai.api_key = self.api_key

    def _validate_chunks(self, chunks: List[ChunkMetadata]) -> None:
        """Validate input chunks.

        Args:
            chunks: List of chunk metadata to validate

        Raises:
            ValueError: If chunks are invalid
        """
        if not chunks:
            raise ValueError("Empty chunk list provided")
        if any(not isinstance(chunk, ChunkMetadata) for chunk in chunks):
            raise ValueError("All chunks must be ChunkMetadata objects")

    async def embed_chunks(self, chunks: List[ChunkMetadata]) -> List[ChunkWithEmbedding]:
        """Embed chunks using OpenAI API.
        
        Args:
            chunks: List of chunks to embed
            
        Returns:
            List of chunks with embeddings
        """
        self._validate_chunks(chunks)
        
        if not chunks:
            return []
            
        try:
            # Extract text from chunks
            texts = [chunk.text for chunk in chunks]
            
            # Get embeddings from OpenAI
            response = openai.Embedding.create(
                input=texts,
                model=self.model_name
            )
            
            # Create ChunkWithEmbedding objects
            chunk_embeddings = []
            for chunk, data in zip(chunks, response["data"]):
                embedding = np.array(data["embedding"])
                chunk_embeddings.append(
                    ChunkWithEmbedding(
                        text=chunk.text,
                        embedding=embedding,
                        char_span=chunk.char_span,
                        token_span=chunk.token_span
                    )
                )
            
            return chunk_embeddings
            
        except Exception as e:
            raise EmbeddingProcessError(f"Failed to get embeddings from OpenAI: {str(e)}") from e
