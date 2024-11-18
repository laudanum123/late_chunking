import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from embedder_late_chunking import LateChunkingEmbedder
import numpy as np
import faiss
import pickle
import argparse
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from dataclasses import dataclass
from embedders import create_embedder, load_config

@dataclass
class Document:
    id: str
    text: str
    metadata: Dict = None

@dataclass
class SearchResult:
    document_id: str
    chunk_text: str
    score: float
    char_start: int = None
    char_end: int = None

class FaissVectorStore:
    def __init__(self, embedding_size: int):
        # Always use cosine similarity
        self.index = faiss.IndexFlatIP(embedding_size)
        self.documents: List[Document] = []
        self.chunks: List[Tuple[str, int, int, str]] = []  # (chunk_text, start, end, doc_id)
        self.duplicate_threshold = 0.95
        self.embedding_size = embedding_size

    def _find_duplicates(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Find duplicate embeddings by checking similarity with existing vectors.
        Returns a boolean numpy array where True indicates a duplicate.
        """
        if len(self.chunks) == 0:  # If store is empty, no duplicates
            return np.zeros(len(embeddings), dtype=bool)
            
        # Search for nearest neighbors among existing embeddings
        k = 1  # We only need the closest match
        scores, _ = self.index.search(embeddings, k)
        
        # Consider vectors with similarity above threshold as duplicates
        is_duplicate = scores[:, 0] > self.duplicate_threshold
        return is_duplicate

    def add_embeddings(self, embeddings: np.ndarray, chunks: List[Tuple[str, int, int, str]]):
        if len(embeddings) == 0:
            return
            
        # Convert embeddings to float32 and ensure correct shape
        embeddings = np.array(embeddings, dtype=np.float32)
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
            
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
            
        # Check for duplicates
        is_duplicate = self._find_duplicates(embeddings)
        
        # Count and log duplicates
        num_duplicates = np.sum(is_duplicate)
        if num_duplicates > 0:
            duplicate_chunks = [chunk[0][:100] + "..." if len(chunk[0]) > 100 else chunk[0] 
                              for chunk, is_dup in zip(chunks, is_duplicate) if is_dup]
            print(f"\nFound {num_duplicates} duplicate embeddings:")
            for chunk in duplicate_chunks:
                print(f"- {chunk}")
        
        # Filter out duplicates
        non_duplicate_mask = ~is_duplicate
        unique_embeddings = embeddings[non_duplicate_mask]
        unique_chunks = [chunk for chunk, not_dup in zip(chunks, non_duplicate_mask) if not_dup]
        
        # Add only unique embeddings
        if len(unique_embeddings) > 0:
            self.index.add(unique_embeddings)
            self.chunks.extend(unique_chunks)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[SearchResult]:
        # Ensure query_embedding is 2D
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid index
                chunk = self.chunks[idx]
                results.append(SearchResult(
                    document_id=chunk[3],
                    chunk_text=chunk[0],
                    score=float(score),
                    char_start=chunk[1],
                    char_end=chunk[2]
                ))
        return results

    def save(self, filepath: str):
        """Save the vector store to disk"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the index
        faiss.write_index(self.index, f"{filepath}.index")
        
        # Save the chunks and other attributes
        state = {
            'chunks': self.chunks,
            'documents': self.documents,
            'duplicate_threshold': self.duplicate_threshold,
            'embedding_size': self.embedding_size
        }
        with open(f"{filepath}.state", 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, filepath: str) -> Optional['FaissVectorStore']:
        """Load the vector store from disk"""
        if not (os.path.exists(f"{filepath}.index") and os.path.exists(f"{filepath}.state")):
            return None
            
        # Load the state
        with open(f"{filepath}.state", 'rb') as f:
            state = pickle.load(f)
            
        # Create instance and restore state
        instance = cls(state['embedding_size'])
        instance.chunks = state['chunks']
        instance.documents = state['documents']
        instance.duplicate_threshold = state['duplicate_threshold']
        
        # Load the index
        instance.index = faiss.read_index(f"{filepath}.index")
        
        return instance

def load_documents_from_directory(directory_path: str) -> List[Document]:
    """
    Load all .txt files from the specified directory as documents.
    
    Args:
        directory_path (str): Path to the directory containing text files
        
    Returns:
        List[Document]: List of Document objects created from text files
    """
    documents = []
    
    # Ensure directory exists
    if not os.path.exists(directory_path):
        raise ValueError(f"Directory not found: {directory_path}")
    
    # Read all .txt files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    # Use filename without extension as document ID
                    doc_id = os.path.splitext(filename)[0]
                    documents.append(Document(id=doc_id, text=text))
            except Exception as e:
                print(f"Error reading file {filename}: {str(e)}")
    
    if not documents:
        raise ValueError(f"No .txt files found in directory: {directory_path}")
    
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

def compare_retrieval_quality(docs_directory: str = "documents", recompute: bool = False, model_name: str = None):
    """
    Compare retrieval quality between late chunking and traditional approaches.
    
    Args:
        docs_directory (str): Directory containing the documents to process
        recompute (bool): If True, recompute all embeddings. If False, try to load from disk.
        model_name (str): Name of the embedding model to use (from config.yaml)
    """
    # Create or open the output file
    with open('comparison_results.txt', 'w', encoding='utf-8') as f:
        f.write("RAG Comparison Results\n")
        f.write("=" * 80 + "\n\n")
        
        print("Initializing embedders...")
        f.write("Initializing embedders...\n")
        
        # Create embedders using config
        late_chunking_embedder = LateChunkingEmbedder(max_length=8000)
        traditional_embedder = create_embedder(model_name)
        
        print(f"Using embedding model: {traditional_embedder.__class__.__name__}")
        f.write(f"Using embedding model: {traditional_embedder.__class__.__name__}\n")

        # Try to load existing vector stores if not recomputing
        late_chunking_store = None
        traditional_store = None
        
        store_suffix = f"_{model_name}" if model_name else ""
        
        if not recompute:
            print("\nTrying to load existing vector stores...")
            late_chunking_store = FaissVectorStore.load(f"vector_stores/late_chunking{store_suffix}")
            traditional_store = FaissVectorStore.load(f"vector_stores/traditional{store_suffix}")

        # Initialize new stores if needed
        if late_chunking_store is None:
            late_chunking_store = FaissVectorStore(late_chunking_embedder.embedding_size)
        if traditional_store is None:
            traditional_store = FaissVectorStore(traditional_embedder.embedding_size)

        # Load documents from directory
        try:
            documents = load_documents_from_directory(docs_directory)
            print(f"\nLoaded {len(documents)} documents from {docs_directory}")
            f.write(f"\nLoaded {len(documents)} documents from {docs_directory}\n")
        except Exception as e:
            print(f"Error loading documents: {str(e)}")
            return

        # Only process documents if recomputing or stores were not loaded
        if recompute or len(late_chunking_store.chunks) == 0:
            print("\nProcessing documents with late chunking approach...")
            f.write("\nProcessing documents with late chunking approach...\n")
            for doc in tqdm(documents):
                chunks_with_embeddings = late_chunking_embedder.embed_text(doc.text)
                embeddings = np.array([c.embedding for c in chunks_with_embeddings])
                chunks_info = [(c.text, c.start_char, c.end_char, doc.id) for c in chunks_with_embeddings]
                late_chunking_store.add_embeddings(embeddings, chunks_info)
            
            # Save the late chunking store
            late_chunking_store.save(f"vector_stores/late_chunking{store_suffix}")

        if recompute or len(traditional_store.chunks) == 0:
            print("\nProcessing documents with traditional approach...")
            f.write("\nProcessing documents with traditional approach...\n")
            for doc in tqdm(documents):
                chunks = traditional_embedder._chunk_text(doc.text)
                embeddings = traditional_embedder.embed_chunks(chunks)
                pos = 0
                chunks_info = []
                for chunk in chunks:
                    start = doc.text.find(chunk.strip("."), pos)
                    if start != -1:
                        chunks_info.append((chunk, start, start + len(chunk), doc.id))
                        pos = start + 1
                traditional_store.add_embeddings(embeddings, chunks_info)
            
            # Save the traditional store
            traditional_store.save(f"vector_stores/traditional{store_suffix}")

        # Collect embeddings and document IDs for visualization from the vector stores
        print("\nPreparing visualization data...")
        late_chunking_embeddings = []
        late_chunking_doc_ids = []
        traditional_embeddings = []
        traditional_doc_ids = []
        
        # Get embeddings from late chunking store
        late_chunking_embeddings = late_chunking_store.index.reconstruct_n(0, late_chunking_store.index.ntotal)
        late_chunking_doc_ids = [chunk[3] for chunk in late_chunking_store.chunks]  # doc_id is at index 3
        
        # Get embeddings from traditional store
        traditional_embeddings = traditional_store.index.reconstruct_n(0, traditional_store.index.ntotal)
        traditional_doc_ids = [chunk[3] for chunk in traditional_store.chunks]  # doc_id is at index 3
            
        # Create visualizations
        print("\nCreating embedding visualizations...")
        visualize_embeddings(
            late_chunking_embeddings,
            late_chunking_doc_ids,
            "Late Chunking Embeddings Visualization",
            f"visualizations/late_chunking_embeddings{store_suffix}.png"
        )
        visualize_embeddings(
            traditional_embeddings,
            traditional_doc_ids,
            "Traditional Embeddings Visualization",
            f"visualizations/traditional_embeddings{store_suffix}.png"
        )
        
        # Test queries targeting different aspects of the text
        test_queries = [
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

        f.write("\nComparing retrieval results:\n")
        f.write("=" * 80 + "\n")
        print("\nComparing retrieval results:")
        print("=" * 80)
        
        for query in test_queries:
            # Write and print query
            query_header = f"\nQuery: {query}"
            f.write(query_header + "\n")
            f.write("-" * 80 + "\n")
            print(query_header)
            print("-" * 80)
            
            # Late chunking results
            query_emb = late_chunking_embedder.embed_query(query)
            late_results = late_chunking_store.search(query_emb, k=10)
            
            # Traditional results
            trad_emb = traditional_embedder.embed_chunks([query])[0]
            trad_results = traditional_store.search(trad_emb, k=10)
            
            # Write and print late chunking results
            f.write("\nLate Chunking Results:\n")
            print("\nLate Chunking Results:")
            for i, r in enumerate(late_results, 1):
                result_line = f"{i}. Score: {r.score:.4f} | {r.chunk_text}"
                f.write(result_line + "\n")
                print(result_line)
            
            # Write and print traditional results
            f.write("\nTraditional Results:\n")
            print("\nTraditional Results:")
            for i, r in enumerate(trad_results, 1):
                result_line = f"{i}. Score: {r.score:.4f} | {r.chunk_text}"
                f.write(result_line + "\n")
                print(result_line)
            
            f.write("=" * 80 + "\n")
            print("=" * 80)

        print("\nResults have been saved to 'comparison_results.txt'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare RAG retrieval quality with different chunking approaches')
    parser.add_argument('--docs_directory', type=str, default="documents",
                      help='Directory containing the documents to process')
    parser.add_argument('--recompute', action='store_true',
                      help='Force recomputation of embeddings instead of loading from disk')
    parser.add_argument('--model', type=str, default=None,
                      help='Name of the embedding model to use (from config.yaml)')
    
    args = parser.parse_args()
    compare_retrieval_quality(args.docs_directory, args.recompute, args.model)
