"""Late chunking embedder implementation."""
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Tuple, Optional
import logging
from pathlib import Path
import faiss
import pickle

from .base import BaseEmbedder, EmbeddingConfig, ModelLoadError, EmbeddingProcessError
from .vector_store import ChunkWithEmbedding

logger = logging.getLogger(__name__)

class LateChunkingEmbedder(BaseEmbedder):
    """Embedder using late chunking strategy."""
    
    def __init__(self, config: EmbeddingConfig):
        """Initialize late chunking embedder.
        
        Args:
            config: EmbeddingConfig instance
        """
        super().__init__(config)
        self.tokenizer = None
        self.model = None
        self.device = None
        self.chunks = []
        self.index = None
        self.batch_size = config.additional_params.get('batch_size', 32) if config.additional_params else 32
        self.vector_store_path = None

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
        inputs = self.tokenizer(input_text, return_tensors='pt', return_offsets_mapping=True)
        punctuation_mark_id = self.tokenizer.convert_tokens_to_ids('.')
        sep_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
        token_offsets = inputs['offset_mapping'][0]
        token_ids = inputs['input_ids'][0]
        
        # Find sentence boundaries
        chunk_positions = [
            (i, int(start + 1))
            for i, (token_id, (start, end)) in enumerate(zip(token_ids, token_offsets))
            if token_id == punctuation_mark_id
            and (
                token_offsets[i + 1][0] - token_offsets[i][1] > 0
                or token_ids[i + 1] == sep_id
            )
        ]
        
        # Create text chunks and their spans
        chunks = [
            input_text[x[1] : y[1]]
            for x, y in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
        ]
        token_spans = [
            (x[0], y[0]) for x, y in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
        ]
        char_spans = [
            (x[1], y[1]) for x, y in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
        ]
        
        return chunks, token_spans, char_spans

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

    async def embed_chunks(self, chunks: List[str]) -> List[ChunkWithEmbedding]:
        """Embed a list of text chunks using late chunking.
        
        Args:
            chunks: List of text chunks to embed
            
        Returns:
            List of chunks with embeddings
        """
        if not chunks:
            return []
            
        # First, process all chunks to get sentence-level chunks
        all_sentence_chunks = []
        for text in chunks:
            # Process text in macro chunks
            sentence_count = 0
            for chunk_text, _ in self._create_macro_chunks(text):
                sentence_chunks, _, _ = self._chunk_by_sentences(chunk_text)
                sentence_count += len(sentence_chunks)
                all_sentence_chunks.extend([c for c in sentence_chunks if c.strip()])
            logger.info(f"Split document into {sentence_count} sentence-level chunks")
            
        # Check if we have these chunks in the vector store
        if self._load_vector_store():
            # Check if all chunks are in the store
            all_found = all(chunk in self.chunks for chunk in all_sentence_chunks)
            if all_found:
                logger.info("Using cached embeddings from vector store")
                chunk_embeddings = []
                for chunk in all_sentence_chunks:
                    idx = self.chunks.index(chunk)
                    embedding = self.index.reconstruct(idx)
                    chunk_embeddings.append(
                        ChunkWithEmbedding(text=chunk, embedding=embedding)
                    )
                return chunk_embeddings
        
        try:
            all_chunk_embeddings = []
            
            # Process each document
            for text in chunks:
                # Process text in macro chunks
                for chunk_text, chunk_start in self._create_macro_chunks(text):
                    # Get sentence chunks and their spans
                    sentence_chunks, token_spans, _ = self._chunk_by_sentences(chunk_text)
                    
                    # Get embeddings for all sentences in this chunk
                    inputs = self.tokenizer(chunk_text, return_tensors='pt').to(self.device)
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    
                    # Create embeddings using late chunking
                    embeddings = self._late_chunking(outputs, token_spans)
                    
                    # Create ChunkWithEmbedding objects for each sentence
                    for chunk, embedding in zip(sentence_chunks, embeddings):
                        if chunk.strip():  # Skip empty chunks
                            # Normalize the embedding
                            normalized_embedding = embedding / np.linalg.norm(embedding)
                            all_chunk_embeddings.append(
                                ChunkWithEmbedding(text=chunk, embedding=normalized_embedding)
                            )
            
            # Add embeddings to vector store
            embeddings_array = np.array([chunk.embedding for chunk in all_chunk_embeddings])
            self._add_embeddings(embeddings_array, all_chunk_embeddings)
            
            return all_chunk_embeddings

        except Exception as e:
            logger.error(f"Error computing embeddings: {str(e)}")
            raise EmbeddingProcessError(f"Failed to compute embeddings: {str(e)}") from e

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
