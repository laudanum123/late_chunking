"""Base embedder module providing abstract base classes and common functionality."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np
import logging
from pathlib import Path
import faiss
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConfig:
    """Configuration for embedding models.
    
    Attributes:
        name: Name or path of the embedding model
        type: Type of embedder (e.g., 'huggingface', 'openai')
        embedding_size: Size of the output embeddings
        max_length: Maximum sequence length
        api_key: Optional API key for cloud services
        additional_params: Optional additional parameters
    """
    name: str
    type: str
    embedding_size: int
    max_length: int
    api_key: Optional[str] = None
    additional_params: Optional[Dict[str, Any]] = None

class EmbeddingError(Exception):
    """Base exception for embedding-related errors."""
    pass

class ModelLoadError(EmbeddingError):
    """Raised when model loading fails."""
    pass

class EmbeddingProcessError(EmbeddingError):
    """Raised when embedding process fails."""
    pass

class BaseEmbedder(ABC):
    """Abstract base class for all embedders."""
    
    def __init__(self, config: EmbeddingConfig):
        """Initialize the embedder with configuration.
        
        Args:
            config: EmbeddingConfig instance containing model settings
            
        Raises:
            ModelLoadError: If model initialization fails
        """
        self.config = config
        self.name = config.name
        self.vector_store_dir = Path("src/late_chunking/vector_store/stores")
        self.vector_store_dir.mkdir(exist_ok=True)
        self.index: Optional[faiss.Index] = None
        self.chunks: List[str] = []
        self.dimension: Optional[int] = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def _get_store_path(self, suffix: str = "") -> Path:
        """Get path for vector store files.
        
        Args:
            suffix: Optional suffix for the filename
            
        Returns:
            Path to vector store file
        """
        base_name = f"{self.name.lower()}{suffix}"
        return self.vector_store_dir / base_name
    
    def _save_vector_store(self) -> None:
        """Save vector store to disk."""
        if self.index is None or not self.chunks:
            logger.warning("No index or chunks to save")
            return
            
        try:
            # Save the FAISS index
            index_path = self._get_store_path(".index")
            faiss.write_index(self.index, str(index_path))
            
            # Save the chunk texts and metadata
            state = {
                'chunks': self.chunks,
                'dimension': self.dimension
            }
            state_path = self._get_store_path(".state")
            with open(state_path, 'wb') as f:
                pickle.dump(state, f)
                
            logger.info(f"Vector store saved to {self.vector_store_dir}")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
    
    def _load_vector_store(self) -> bool:
        """Load vector store from disk.
        
        Returns:
            bool: True if successfully loaded
        """
        index_path = self._get_store_path(".index")
        state_path = self._get_store_path(".state")
        
        if not index_path.exists() or not state_path.exists():
            logger.info("No existing vector store found")
            return False
            
        try:
            # Load the FAISS index
            self.index = faiss.read_index(str(index_path))
            
            # Load the chunk texts and metadata
            with open(state_path, 'rb') as f:
                state = pickle.load(f)
            self.chunks = state['chunks']
            self.dimension = state['dimension']
            
            logger.info(f"Vector store loaded from {self.vector_store_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False
    
    def _init_index(self, dimension: int) -> None:
        """Initialize FAISS index.
        
        Args:
            dimension: Embedding dimension
        """
        if self.index is None:
            self.dimension = dimension
            self.index = faiss.IndexFlatL2(dimension)
            
    def _add_embeddings(self, embeddings: np.ndarray, texts: List[str]) -> None:
        """Add embeddings to index.
        
        Args:
            embeddings: Embedding vectors
            texts: Corresponding text chunks
        """
        if self.index is None:
            self._init_index(embeddings.shape[1])
            
        self.index.add(embeddings)
        self.chunks.extend(texts)
        self._save_vector_store()
    
    @abstractmethod
    async def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        """Embed a list of text chunks asynchronously.
        
        Args:
            chunks: List of text chunks to embed
            
        Returns:
            numpy.ndarray: Array of embeddings
            
        Raises:
            EmbeddingProcessError: If embedding process fails
        """
        pass
    
    async def embed_query(self, query: str) -> np.ndarray:
        """Embed a query string without adding to vector store.
        
        Args:
            query: Query string to embed
            
        Returns:
            Query embedding as numpy array
        """
        raise NotImplementedError("Subclasses must implement embed_query")

    def _validate_chunks(self, chunks: List[str]) -> None:
        """Validate input chunks.
        
        Args:
            chunks: List of text chunks to validate
            
        Raises:
            ValueError: If chunks are invalid
        """
        if not chunks:
            raise ValueError("Empty chunk list provided")
        if any(not isinstance(chunk, str) for chunk in chunks):
            raise ValueError("All chunks must be strings")
            
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings.
        
        Args:
            embeddings: Input embeddings array
            
        Returns:
            numpy.ndarray: Normalized embeddings
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms
        
    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass

    def __enter__(self):
        """Context manager entry."""
        raise TypeError("Use 'async with' instead")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        raise TypeError("Use 'async with' instead")

    def cleanup(self):
        """Cleanup resources. Override in subclasses if needed."""
        pass
