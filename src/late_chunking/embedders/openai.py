"""OpenAI-based embedder implementation."""
import openai
import numpy as np
from typing import List, Optional
import os
import asyncio
import logging
from pathlib import Path

from .base import BaseEmbedder, EmbeddingConfig, ModelLoadError, EmbeddingProcessError

logger = logging.getLogger(__name__)

class OpenAIEmbedder(BaseEmbedder):
    """Embedder implementation using OpenAI's API."""
    
    def __init__(self, config: EmbeddingConfig):
        """Initialize OpenAI embedder.
        
        Args:
            config: EmbeddingConfig instance
            
        Raises:
            ModelLoadError: If initialization fails
        """
        super().__init__(config)
        try:
            # Try to get API key from config first, then environment
            api_key = config.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found in config or environment")
            openai.api_key = api_key
            logger.info("OpenAI API configured successfully")
        except Exception as e:
            raise ModelLoadError(f"Failed to initialize OpenAI embedder: {str(e)}") from e

    async def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        """Embed text chunks asynchronously using OpenAI's API.
        
        Args:
            chunks: List of text chunks to embed
            
        Returns:
            numpy.ndarray: Array of embeddings
            
        Raises:
            EmbeddingProcessError: If embedding process fails
        """
        self._validate_chunks(chunks)
        try:
            embeddings = []
            for chunk in chunks:
                # Truncate text if needed
                if len(chunk) > self.config.max_length:
                    chunk = chunk[:self.config.max_length]
                    logger.warning(f"Chunk truncated to {self.config.max_length} characters")
                
                # Allow other async operations to proceed
                await asyncio.sleep(0)
                
                response = await self._get_embedding(chunk)
                embedding = response['data'][0]['embedding']
                embeddings.append(embedding)
            
            embeddings = np.array(embeddings, dtype=np.float32)
            return self._normalize_embeddings(embeddings)
            
        except Exception as e:
            raise EmbeddingProcessError(f"OpenAI embedding failed: {str(e)}") from e
    
    async def _get_embedding(self, text: str) -> dict:
        """Get embeddings from OpenAI API with retry logic.
        
        Args:
            text: Text to embed
            
        Returns:
            dict: OpenAI API response
            
        Raises:
            EmbeddingProcessError: If API call fails after retries
        """
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                return await openai.Embedding.acreate(
                    input=text,
                    model=self.config.name
                )
            except Exception as e:
                if attempt == max_retries - 1:
                    raise EmbeddingProcessError(f"OpenAI API failed after {max_retries} attempts: {str(e)}")
                logger.warning(f"Retry {attempt + 1}/{max_retries} after error: {str(e)}")
                await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
