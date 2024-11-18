"""HuggingFace-based embedder implementation."""
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List
from pathlib import Path
import logging
import faiss
import pickle

from .base import BaseEmbedder, EmbeddingConfig, ModelLoadError, EmbeddingProcessError
from .vector_store import ChunkWithEmbedding

logger = logging.getLogger(__name__)

class HuggingFaceEmbedder(BaseEmbedder):
    """Embedder implementation using HuggingFace models."""
    
    def __init__(self, config: EmbeddingConfig):
        """Initialize HuggingFace embedder.
        
        Args:
            config: EmbeddingConfig instance
        """
        super().__init__(config)
        self.tokenizer = None
        self.model = None
        self.device = None
        self.chunks = []
        self.index = None
        self.vector_store_path = None
        self.batch_size = config.additional_params.get('batch_size', 32) if config.additional_params else 32

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
            logger.info(f"Model loaded successfully on {self.device}")
            return self
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {str(e)}") from e

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if hasattr(self, 'model'):
            self.model.cpu()
            torch.cuda.empty_cache()
            logger.info("Model resources cleaned up")

    def set_vector_store_path(self, path: Path):
        """Set the path for vector store operations."""
        self.vector_store_path = path

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

    def _chunk_by_sentences(self, input_text: str) -> List[str]:
        """Split text into sentences.
        
        Args:
            input_text: Text to split into sentences
            
        Returns:
            List of sentence chunks
        """
        # Tokenize the text
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False
        )
        
        token_ids = inputs["input_ids"][0]
        token_offsets = inputs["offset_mapping"][0]
        
        # Get the separator token id (usually period)
        sep_id = self.tokenizer.convert_tokens_to_ids(["."])[0]
        
        # Find chunk positions based on sentence boundaries
        chunk_positions = [
            (token_offsets[i][0].item(), token_offsets[i][1].item())
            for i in range(len(token_ids) - 1)
            if token_ids[i] == sep_id
            and (
                token_offsets[i + 1][0] - token_offsets[i][1] > 0
                or token_ids[i + 1] == sep_id
            )
        ]
        
        # Add the last position if it's not a separator
        if token_ids[-1] != sep_id:
            chunk_positions.append(
                (token_offsets[-1][0].item(), token_offsets[-1][1].item())
            )
        
        # Create text chunks
        chunks = [
            input_text[x[1]:y[1]].strip()
            for x, y in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
        ]
        
        # Filter out empty chunks
        chunks = [chunk for chunk in chunks if chunk]
        
        return chunks

    async def embed_chunks(self, chunks: List[str]) -> List[ChunkWithEmbedding]:
        """Embed a list of text chunks asynchronously.
        
        Args:
            chunks: List of text chunks to embed
            
        Returns:
            List of chunks with embeddings
        """
        try:
            all_chunks = []
            for text in chunks:
                sentence_chunks = self._chunk_by_sentences(text)
                all_chunks.extend([c for c in sentence_chunks if c.strip()])
            
            # Check if we have these chunks in the vector store
            if self._load_vector_store():
                # Check if all chunks are in the store
                all_found = all(chunk in self.chunks for chunk in all_chunks)
                if all_found:
                    logger.info("Using cached embeddings from vector store")
                    chunk_embeddings = []
                    for chunk in all_chunks:
                        idx = self.chunks.index(chunk)
                        embedding = self.index.reconstruct(idx)
                        chunk_embeddings.append(
                            ChunkWithEmbedding(text=chunk, embedding=embedding)
                        )
                    return chunk_embeddings
        
            # If not in vector store, compute embeddings
            try:
                # Process in batches
                embeddings = []
                
                for i in range(0, len(all_chunks), self.batch_size):
                    batch = all_chunks[i:i + self.batch_size]
                    
                    # Tokenize and get embeddings
                    inputs = self.tokenizer(
                        batch,
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
                    batch_embeddings = (embeddings_sum / mask_sum).cpu().numpy()
                    
                    # Normalize embeddings
                    batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                    embeddings.extend(batch_embeddings)
                
                # Create ChunkWithEmbedding objects
                chunk_embeddings = [
                    ChunkWithEmbedding(text=chunk, embedding=embedding)
                    for chunk, embedding in zip(all_chunks, embeddings)
                ]
                
                # Add embeddings to vector store
                embeddings_array = np.array([e for e in embeddings])
                self._add_embeddings(embeddings_array, chunk_embeddings)
                
                return chunk_embeddings
                
            except Exception as e:
                raise EmbeddingProcessError(f"Failed to embed chunks: {str(e)}") from e

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
