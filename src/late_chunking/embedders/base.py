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

@dataclass
class ChunkWithEmbedding:
    """A text chunk with its embedding.
    
    Attributes:
        text: The text content of the chunk
        embedding: The embedding vector for the chunk
        char_span: Character span (start, end) in the original text
        token_span: Token span (start, end) in the original text
    """
    text: str
    embedding: np.ndarray
    char_span: Optional[tuple[int, int]] = None
    token_span: Optional[tuple[int, int]] = None

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
    
    def __init__(self, config: EmbeddingConfig, vector_store_dir: Optional[str] = None):
        """Initialize the embedder with configuration.
        
        Args:
            config: EmbeddingConfig instance containing model settings
            vector_store_dir: Optional path to vector store directory. If not provided,
                            defaults to "src/late_chunking/vector_store/stores"
            
        Raises:
            ModelLoadError: If model initialization fails
        """
        self.config = config
        self.name = config.name
        self.vector_store_dir = Path(vector_store_dir or "src/late_chunking/vector_store/stores")
        self.vector_store_path = self.vector_store_dir  # Initialize to same as vector_store_dir
        self.vector_store_dir.mkdir(exist_ok=True, parents=True)
        self.index: Optional[faiss.Index] = None
        self.chunks: List[str] = []
        self.dimension: Optional[int] = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def set_vector_store_path(self, path: Path) -> None:
        """Set the path for vector store operations.
        
        Args:
            path: New path for vector store operations
        """
        self.vector_store_path = path
        self.vector_store_path.mkdir(exist_ok=True, parents=True)
    
    def _save_vector_store(self):
        """Save vector store to disk."""
        try:
            # Verify index and chunks are in sync before saving
            if self.index is None or len(self.chunks) == 0:
                logger.warning("No embeddings to save")
                return
                
            if self.index.ntotal != len(self.chunks):
                logger.error(f"Cannot save vector store: index has {self.index.ntotal} vectors but chunks list has {len(self.chunks)} items")
                raise ValueError("Index and chunks list are out of sync")
            
            # Create parent directory if it doesn't exist
            self.vector_store_path.mkdir(parents=True, exist_ok=True)
            
            # Save the FAISS index
            index_path = self.vector_store_path / "index.faiss"
            faiss.write_index(self.index, str(index_path))
            
            # Save the chunks and metadata
            chunks_path = self.vector_store_path / "chunks.pkl"
            with open(chunks_path, 'wb') as f:
                # Convert chunks to serializable format
                chunk_data = [
                    {
                        'text': chunk.text if hasattr(chunk, 'text') else str(chunk),
                        'embedding': chunk.embedding if hasattr(chunk, 'embedding') else None,
                        'char_span': chunk.char_span if hasattr(chunk, 'char_span') else None,
                        'token_span': chunk.token_span if hasattr(chunk, 'token_span') else None,
                        'doc_id': chunk.doc_id if hasattr(chunk, 'doc_id') else None
                    }
                    for chunk in self.chunks
                ]
                pickle.dump({
                    'chunks': chunk_data,
                    'dimension': self.dimension,
                    'total_vectors': self.index.ntotal
                }, f)
                
            logger.info(f"Vector store saved to {self.vector_store_path} with {self.index.ntotal} vectors and {len(self.chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise

    def _load_vector_store(self) -> bool:
        """Load vector store from disk.
        
        Returns:
            bool: True if successfully loaded
        """
        try:
            if not self.vector_store_path:
                logger.warning("No vector store path set")
                return False
                
            index_path = self.vector_store_path / "index.faiss"
            chunks_path = self.vector_store_path / "chunks.pkl"
            
            if not index_path.exists() or not chunks_path.exists():
                logger.info("No existing vector store found")
                return False
                
            # Load the FAISS index first to get embeddings
            self.index = faiss.read_index(str(index_path))
            
            # Load chunks and metadata
            with open(chunks_path, 'rb') as f:
                data = pickle.load(f)
                chunk_data = data['chunks']
                self.dimension = data['dimension']
                expected_vectors = data['total_vectors']
                
                # Convert loaded data back to ChunkWithEmbedding objects
                self.chunks = []
                for i, chunk in enumerate(chunk_data):
                    if isinstance(chunk, dict):
                        # Get embedding from FAISS index if not in chunk data
                        if chunk['embedding'] is None and i < self.index.ntotal:
                            embedding = np.zeros((1, self.dimension), dtype=np.float32)
                            self.index.reconstruct(i, embedding.reshape(-1))
                            chunk['embedding'] = embedding.reshape(-1)
                            
                        chunk_obj = ChunkWithEmbedding(
                            text=chunk['text'],
                            embedding=chunk['embedding'],
                            char_span=chunk['char_span'],
                            token_span=chunk['token_span']
                        )
                        if chunk.get('doc_id') is not None:
                            chunk_obj.doc_id = chunk['doc_id']
                        self.chunks.append(chunk_obj)
                    else:
                        # Handle legacy format or plain strings
                        if i < self.index.ntotal:
                            embedding = np.zeros((1, self.dimension), dtype=np.float32)
                            self.index.reconstruct(i, embedding.reshape(-1))
                            chunk_obj = ChunkWithEmbedding(
                                text=str(chunk),
                                embedding=embedding.reshape(-1)
                            )
                            self.chunks.append(chunk_obj)
                        else:
                            logger.warning(f"Skipping chunk {i} as it has no embedding")
            
            # Verify loaded data
            if self.index.ntotal != expected_vectors:
                logger.error(f"Index corruption detected: saved index had {expected_vectors} vectors but loaded index has {self.index.ntotal}")
                return False
                
            if self.index.ntotal != len(self.chunks):
                logger.error(f"Data corruption: index has {self.index.ntotal} vectors but chunks list has {len(self.chunks)} items")
                return False
            
            logger.info(f"Vector store loaded from {self.vector_store_path} with {self.index.ntotal} vectors")
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
            
    def _add_embeddings(self, embeddings_array, chunks):
        """Add new embeddings to vector store.
        
        Args:
            embeddings_array: numpy array of embeddings to add
            chunks: list of chunks corresponding to the embeddings
        """
        try:
            # Initialize or reset the index if needed
            if self.index is None:
                self.dimension = embeddings_array.shape[1]
                self.index = faiss.IndexFlatL2(self.dimension)
                self.chunks = []  # Reset chunks when creating new index
            
            # Add embeddings to index
            self.index.add(embeddings_array)
            
            # Update chunks list
            self.chunks.extend(chunks)
            
            # Verify index and chunks are in sync
            if self.index.ntotal != len(self.chunks):
                logger.error(f"Index-chunks mismatch after adding embeddings: index has {self.index.ntotal} vectors but chunks list has {len(self.chunks)} items")
                raise ValueError("Index and chunks list are out of sync")
                
        except Exception as e:
            logger.error(f"Error adding to vector store: {str(e)}")
            raise
    
    @abstractmethod
    async def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
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

    def _validate_chunks(self, chunks: List[Any]) -> None:
        """Validate input chunks.
        
        Args:
            chunks: List of chunks to validate
            
        Raises:
            ValueError: If chunks are invalid
        """
        if not chunks:
            raise ValueError("Empty chunk list provided")
        if any(not isinstance(chunk, ChunkWithEmbedding) for chunk in chunks):
            raise ValueError("All chunks must be ChunkWithEmbedding objects")
            
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
