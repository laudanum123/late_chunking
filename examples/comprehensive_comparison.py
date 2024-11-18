"""Comprehensive RAG comparison example showing retrieved chunks."""
import asyncio
import os
from pathlib import Path
import numpy as np
from typing import List, Dict
import logging
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from late_chunking.rag_comparison import RAGComparison

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample questions about the cities
QUESTIONS = [
    "What is the current population of New York City and how does it compare to other US cities?",
    "How did New York City get its name and what was it called before?",
    "What role does Wall Street play in New York's economy and global finance?",
    "What are the five boroughs of New York City and how are they different?",
    "How does New York City's public transportation system work, especially the subway?",
    "What is the significance of Central Park and when was it established?",
    "How did the September 11, 2001 attacks impact New York City?",
    "What makes New York City a global cultural capital, especially in terms of arts and entertainment?",
    "What are the major tourist attractions in New York City and how many visitors does it receive annually?",
    "How has immigration shaped New York City's demographics and culture?",
    "What is the history of Broadway theater and its impact on New York's culture?",
    "How does New York City's climate vary throughout the year?",
    "What are the major sports teams in New York City and where do they play?",
    "How has New York's architecture evolved, especially its famous skyscrapers?",
    "What role does education play in New York City, including its universities and schools?",
    "How does New York City handle environmental challenges and sustainability?",
    "What is unique about New York City's food culture and cuisine?",
    "How does the New York City government structure work?",
    "What are the major bridges and tunnels connecting New York's boroughs?",
    "How has technology and Silicon Alley contributed to New York's economy?",
    "What is the current role of Bonn in German government?",
    "Tell me about Bonn's history as a capital city",
    "What international organizations are located in Bonn?",
    "How can I visit Beethoven's birthplace?",
    "What are the main transportation options in Bonn?",
    "What is the significance of the University of Bonn?",
    "What major companies are headquartered in Bonn?",
    "How is Bonn connected to the Rhine river?",
    "What museums can I visit in Bonn?",
    "How did Bonn's role change after German reunification?",
    "What is the climate like in Bonn?",
    "How old is the city of Bonn?",
    "What role did Bonn play during Roman times?",
    "What cultural attractions can tourists visit in Bonn?",
    "How is public transportation organized in Bonn?",
    "What is special about Bonn's educational institutions?",
    "What UN organizations are based in Bonn?",
    "How did Bonn become the capital of West Germany?",
    "What are the main leisure areas in Bonn?",
    "What is Bonn's connection to Beethoven?"
]

def load_documents() -> Dict[str, str]:
    """Load documents from the documents directory.
    
    Returns:
        Dict mapping document names to their contents
    """
    docs_dir = Path(__file__).parent.parent / "documents"
    documents = {}
    
    for doc_file in docs_dir.glob("*.txt"):
        with open(doc_file, "r", encoding="utf-8") as f:
            documents[doc_file.stem] = f.read()
    
    return documents

def visualize_embeddings(embeddings: np.ndarray, doc_ids: List[str], title: str, output_file: str):
    """
    Create a 2D visualization of embeddings using t-SNE.
    
    Args:
        embeddings (np.ndarray): Matrix of embeddings
        doc_ids (List[str]): List of document IDs corresponding to each embedding
        title (str): Title for the plot
        output_file (str): Path to save the visualization
    """
    # Create t-SNE reducer
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    
    # Reduce dimensionality to 2D
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    # Create a mapping of unique document IDs to colors
    unique_docs = list(set(doc_ids))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_docs)))
    doc_to_color = dict(zip(unique_docs, colors))
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot points colored by document
    for doc_id in unique_docs:
        mask = [d == doc_id for d in doc_ids]
        points = reduced_embeddings[mask]
        plt.scatter(points[:, 0], points[:, 1], 
                   c=[doc_to_color[doc_id]], 
                   label=doc_id,
                   alpha=0.6)
    
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

async def main():
    """Run comprehensive RAG comparison."""
    try:
        # Ensure we're in the project root directory
        os.chdir(Path(__file__).parent.parent)
        
        # Create vector store directories
        vector_store_dir = Path("vector_stores") / "comprehensive_comparison"
        vector_store_dir.mkdir(parents=True, exist_ok=True)
        
        # Load documents
        documents = load_documents()
        logger.info(f"Loaded {len(documents)} documents")
        
        # Initialize comparison
        async with RAGComparison() as comparison:
            # Set vector store paths for embedders
            comparison.late_embedder.set_vector_store_path(vector_store_dir / "late_chunking")
            comparison.trad_embedder.set_vector_store_path(vector_store_dir / "traditional")
            
            # Try to load existing vector stores
            late_store_loaded = comparison.late_embedder._load_vector_store()
            trad_store_loaded = comparison.trad_embedder._load_vector_store()
            if late_store_loaded and trad_store_loaded:
                logger.info("Successfully loaded existing vector stores")
                # Extract embeddings from loaded stores
                trad_embeddings = comparison.trad_embedder.chunks
                late_embeddings = comparison.late_embedder.chunks
            else:
                logger.info("Computing new embeddings...")
                # Get all embeddings first
                trad_embeddings, late_embeddings = await comparison._embed_documents(
                    list(documents.values()),
                    list(documents.keys())  # Use document names as IDs
                )
            
            # Create output directory for detailed results
            output_dir = Path("output/detailed_comparison")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process all questions at once
            logger.info(f"Processing {len(QUESTIONS)} questions...")
            
            # Process queries using the embeddings
            results = []
            for query in QUESTIONS:
                result = await comparison._process_query(query, trad_embeddings, late_embeddings)
                results.append(result)
            
            # Extract document IDs first, then convert embeddings for visualization
            traditional_doc_ids = [chunk.doc_id for chunk in trad_embeddings]
            late_doc_ids = [chunk.doc_id for chunk in late_embeddings]
            
            # Now convert embeddings to numpy arrays
            traditional_embeddings = np.array([chunk.embedding for chunk in trad_embeddings])
            late_embeddings = np.array([chunk.embedding for chunk in late_embeddings])
            
            # Log chunk counts for verification
            logger.info(f"Traditional chunks: {len(traditional_embeddings)} from {len(set(traditional_doc_ids))} documents")
            logger.info(f"Late chunking chunks: {len(late_embeddings)} from {len(set(late_doc_ids))} documents")
            
            # Create visualizations
            visualize_embeddings(
                traditional_embeddings,
                traditional_doc_ids,
                "Traditional Chunking Embedding Space",
                output_dir / "traditional_embeddings.png"
            )
            
            visualize_embeddings(
                late_embeddings,
                late_doc_ids,
                "Late Chunking Embedding Space",
                output_dir / "late_embeddings.png"
            )
            
            logger.info("Created embedding visualizations")
            
            # Save vector stores if they've been modified
            comparison.late_embedder.save_vector_store()
            comparison.trad_embedder.save_vector_store()
            logger.info("Saved vector stores")
            
            # Save all results to a single file
            output_file = output_dir / "comparison_results.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("RAG Comparison Results\n")
                f.write("=" * 80 + "\n\n")
                
                # Write summary statistics
                f.write("Summary Statistics\n")
                f.write("-" * 80 + "\n")
                trad_scores = [r.traditional_score for r in results]
                late_scores = [r.late_score for r in results]
                f.write(f"Average Traditional Score: {np.mean(trad_scores):.4f}\n")
                f.write(f"Average Late Chunking Score: {np.mean(late_scores):.4f}\n")
                f.write(f"Traditional Score Range: {min(trad_scores):.4f} - {max(trad_scores):.4f}\n")
                f.write(f"Late Chunking Score Range: {min(late_scores):.4f} - {max(late_scores):.4f}\n\n")
                
                # Write detailed results for each query
                f.write("Detailed Results\n")
                f.write("=" * 80 + "\n\n")
                for i, result in enumerate(results, 1):
                    f.write(f"Question {i}/{len(results)}\n")
                    f.write(str(result))
                    f.write("\n\n" + "=" * 80 + "\n\n")
            
            logger.info(f"All results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        exit(1)
