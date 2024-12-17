"""RAG comparison module for evaluating different chunking strategies."""
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import yaml

from .embedders import (
    EmbeddingConfig,
    HuggingFaceEmbedder,
    LateChunkingEmbedder,
    ChunkWithEmbedding
)

logger = logging.getLogger(__name__)

class ComparisonResult:
    """Results from comparing traditional and late chunking approaches."""
    
    def __init__(
            self,
            query: str,
            traditional_chunks: List[ChunkWithEmbedding],
            late_chunks: List[ChunkWithEmbedding],
            traditional_score: float,
            late_score: float,
            traditional_chunk_scores: List[float],
            late_chunk_scores: List[float]
        ):
        """Initialize comparison result.
        
        Args:
            query: The query string
            traditional_chunks: Top chunks from traditional approach
            late_chunks: Top chunks from late chunking approach
            traditional_score: Average score for traditional approach
            late_score: Average score for late chunking approach
            traditional_chunk_scores: Individual scores for traditional chunks
            late_chunk_scores: Individual scores for late chunks
        """
        self.query = query
        self.traditional_chunks = traditional_chunks
        self.late_chunks = late_chunks
        self.traditional_score = traditional_score
        self.late_score = late_score
        self.traditional_chunk_scores = traditional_chunk_scores
        self.late_chunk_scores = late_chunk_scores

    def __str__(self) -> str:
        """Format comparison result as string."""
        output = []
        output.append(f"Query: {self.query}")
        output.append("=" * 80)
        output.append("\nScores:")
        output.append(f"Traditional Score: {self.traditional_score:.4f}")
        output.append(f"Late Chunking Score: {self.late_score:.4f}")
        output.append("\n" + "=" * 80)
        
        output.append("\nTraditional Chunking - Top Chunks:")
        output.append("-" * 80)
        for i, (chunk, score) in enumerate(zip(self.traditional_chunks, self.traditional_chunk_scores), 1):
            output.append(f"\n{i}. Score: {score:.4f}")
            output.append(f"Document: {chunk.doc_id}")
            output.append(f"Text: {chunk.text}")
        
        output.append("\nLate Chunking - Top Chunks:")
        output.append("-" * 80)
        for i, (chunk, score) in enumerate(zip(self.late_chunks, self.late_chunk_scores), 1):
            output.append(f"\n{i}. Score: {score:.4f}")
            output.append(f"Document: {chunk.doc_id}")
            output.append(f"Text: {chunk.text}")
        
        return "\n".join(output)

class RAGComparison:
    """Compare traditional and late chunking RAG approaches."""
    
    def __init__(
            self,
            config_path: str = "config.yaml",
            output_dir: Optional[str] = None,
            vector_store_paths: Optional[Dict[str, str]] = None
        ):
        """Initialize RAG comparison.
        
        Args:
            config_path: Path to config file
            output_dir: Optional output directory for storing results and logs
            vector_store_paths: Optional dictionary mapping 'late_chunking' and 'traditional' to vector store paths
        """
        self.config = self._load_config(config_path)
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/default")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_paths = vector_store_paths or {}
        
        # Setup logging
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "comparison.log"),
                logging.StreamHandler()
            ]
        )

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path) as f:
            return yaml.safe_load(f)

    async def __aenter__(self):
        """Create embedder instances."""
        config = EmbeddingConfig(
            name=self.config['model']['name'],
            type=self.config['model']['type'],
            embedding_size=self.config['model']['embedding_size'],
            max_length=self.config['model']['max_length'],
            additional_params=self.config['model'].get('additional_params', {})
        )
        
        self.trad_embedder = await HuggingFaceEmbedder(
            config, 
            vector_store_dir=self.vector_store_paths.get('traditional')
        ).__aenter__()
        self.late_embedder = await LateChunkingEmbedder(
            config,
            vector_store_dir=self.vector_store_paths.get('late_chunking')
        ).__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources."""
        if self.trad_embedder:
            await self.trad_embedder.__aexit__(exc_type, exc_val, exc_tb)
        if self.late_embedder:
            await self.late_embedder.__aexit__(exc_type, exc_val, exc_tb)

    async def _embed_documents_late(
        self,
        documents: List[str],
        doc_ids: List[str]
    ) -> List[ChunkWithEmbedding]:
        """Embed documents using late chunking approach.
        
        Args:
            documents: List of documents to search
            doc_ids: List of document IDs
            
        Returns:
            List of chunks with embeddings
        """
        logger.info("Embedding documents with late chunking approach...")
        late_embeddings = []
        for doc, doc_id in zip(documents, doc_ids):
            logger.info(f"Processing document: {doc_id}")
            # First chunk the document
            chunk_metadata = await self.late_embedder.chunk_text(doc)
            # Then embed the chunks
            chunks = await self.late_embedder.embed_chunks(chunk_metadata)
            for chunk in chunks:
                chunk.doc_id = doc_id
            late_embeddings.extend(chunks)
        return late_embeddings

    async def _embed_documents_traditional(
        self,
        documents: List[str],
        doc_ids: List[str]
    ) -> List[ChunkWithEmbedding]:
        """Embed documents using traditional approach.
        
        Args:
            documents: List of documents to search
            doc_ids: List of document IDs
            
        Returns:
            List of chunks with embeddings
        """
        logger.info("Embedding documents with traditional approach...")
        trad_embeddings = []
        for doc, doc_id in zip(documents, doc_ids):
            logger.info(f"Processing document: {doc_id}")
            # First chunk the document
            chunk_metadata = await self.trad_embedder.chunk_text(doc)
            # Then embed the chunks
            chunks = await self.trad_embedder.embed_chunks(chunk_metadata)
            for chunk in chunks:
                chunk.doc_id = doc_id
            trad_embeddings.extend(chunks)
        return trad_embeddings

    async def _process_query(
            self,
            query: str,
            trad_embeddings: List[ChunkWithEmbedding],
            late_embeddings: List[ChunkWithEmbedding],
            top_k: int = 10
        ) -> ComparisonResult:
        """Process a single query against both sets of embeddings.
        
        Args:
            query: Query string
            trad_embeddings: Traditional approach embeddings
            late_embeddings: Late chunking approach embeddings
            top_k: Number of top chunks to return
            
        Returns:
            ComparisonResult for this query
        """
        # Get query embedding
        query_embedding = await self.trad_embedder.embed_query(query)
        
        # Get top chunks and scores for traditional approach
        trad_scores = []
        for chunk in trad_embeddings:
            # Ensure we're using normalized vectors for cosine similarity
            query_norm = np.linalg.norm(query_embedding)
            chunk_norm = np.linalg.norm(chunk.embedding)
            if query_norm == 0 or chunk_norm == 0:
                score = 0.0
            else:
                score = float(np.dot(query_embedding, chunk.embedding) / (query_norm * chunk_norm))
            trad_scores.append((score, chunk))
        
        # Sort by score
        trad_scores = sorted(trad_scores, key=lambda x: x[0], reverse=True)
        trad_top_chunks = [chunk for _, chunk in trad_scores[:top_k]]
        trad_top_scores = [score for score, _ in trad_scores[:top_k]]
        trad_avg_score = float(np.mean(trad_top_scores))
        
        # Get top chunks and scores for late chunking approach
        late_scores = []
        for chunk in late_embeddings:
            # Ensure we're using normalized vectors for cosine similarity
            query_norm = np.linalg.norm(query_embedding)
            chunk_norm = np.linalg.norm(chunk.embedding)
            if query_norm == 0 or chunk_norm == 0:
                score = 0.0
            else:
                score = float(np.dot(query_embedding, chunk.embedding) / (query_norm * chunk_norm))
            late_scores.append((score, chunk))
        
        # Sort by score
        late_scores = sorted(late_scores, key=lambda x: x[0], reverse=True)
        late_top_chunks = [chunk for _, chunk in late_scores[:top_k]]
        late_top_scores = [score for score, _ in late_scores[:top_k]]
        late_avg_score = float(np.mean(late_top_scores))
        
        return ComparisonResult(
            query=query,
            traditional_chunks=trad_top_chunks,
            late_chunks=late_top_chunks,
            traditional_score=trad_avg_score,
            late_score=late_avg_score,
            traditional_chunk_scores=trad_top_scores,
            late_chunk_scores=late_top_scores
        )

    async def run_comparison(
            self,
            queries: List[str],
            documents: List[str],
            doc_ids: List[str]
        ) -> List[ComparisonResult]:
        """Run RAG comparison.
        
        Args:
            queries: List of queries to test
            documents: List of documents to search
            doc_ids: List of document IDs
            
        Returns:
            List of ComparisonResults
        """
        logger.info("Starting RAG comparison")
        
        # First embed all documents with both approaches
        trad_embeddings = await self._embed_documents_traditional(documents, doc_ids)
        late_embeddings = await self._embed_documents_late(documents, doc_ids)
        
        # Save vector stores after all documents are processed
        logger.info("Saving vector stores...")
        self.trad_embedder.save_vector_store()
        self.late_embedder.save_vector_store()
        
        # Process each query
        results = []
        for query in queries:
            logger.info(f"Processing query: {query}")
            result = await self._process_query(query, trad_embeddings, late_embeddings)
            results.append(result)
        
        return results

async def main():
    """Run RAG comparison from command line."""
    # Example usage
    async with RAGComparison() as comparison:
        queries = [
            "Tell me about the history of New York City",
            "What are the major landmarks in Manhattan?",
            "How does the subway system work?"
        ]
        documents = [
            "New York City was founded in 1624...",
            "The Empire State Building is an iconic landmark...",
            "The NYC subway system is one of the oldest..."
        ]
        doc_ids = ["doc1", "doc2", "doc3"]
        
        results = await comparison.run_comparison(queries, documents, doc_ids)
        for result in results:
            print(result)

if __name__ == "__main__":
    asyncio.run(main())
