import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from embedder_late_chunking import LateChunkingEmbedder
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
import faiss
import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

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

class TraditionalEmbedder:
    def __init__(self, model_name: str = 'jinaai/jina-embeddings-v2-base-en'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        # Get embedding size from model config
        self.embedding_size = self.model.config.hidden_size

    def _chunk_text(self, text: str) -> List[str]:
        """Use same sentence-based chunking as LateChunkingEmbedder"""
        inputs = self.tokenizer(text, return_tensors='pt', return_offsets_mapping=True)
        punctuation_mark_id = self.tokenizer.convert_tokens_to_ids('.')
        sep_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
        token_offsets = inputs['offset_mapping'][0]
        token_ids = inputs['input_ids'][0]
        
        # Find sentence boundaries and track both start and end positions
        chunk_boundaries = []
        current_start = 0
        
        for i, (token_id, (start, end)) in enumerate(zip(token_ids, token_offsets)):
            if token_id == punctuation_mark_id and (
                i + 1 < len(token_offsets) and (
                    token_offsets[i + 1][0] - token_offsets[i][1] > 0
                    or token_ids[i + 1] == sep_id
                )
            ):
                chunk_boundaries.append((current_start, end))
                if i + 1 < len(token_offsets):
                    current_start = token_offsets[i + 1][0]
        
        # Add final chunk if there's remaining text
        if current_start < token_offsets[-1][1]:
            chunk_boundaries.append((current_start, token_offsets[-1][1]))
            
        # Create non-overlapping chunks using the boundaries
        chunks = [text[start:end] for start, end in chunk_boundaries]
        return chunks

    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        embeddings = []
        for chunk in chunks:
            inputs = self.tokenizer(chunk, return_tensors='pt', padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pooling
                embedding = outputs[0][0].mean(dim=0)
                # L2 normalize
                embedding = embedding / torch.norm(embedding)
                embeddings.append(embedding.cpu().numpy())
        return np.array(embeddings)

class FaissVectorStore:
    def __init__(self, embedding_size: int):
        # Always use cosine similarity
        self.index = faiss.IndexFlatIP(embedding_size)
        self.documents: List[Document] = []
        self.chunks: List[Tuple[str, int, int, str]] = []  # (chunk_text, start, end, doc_id)
        self.duplicate_threshold = 0.95

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

class ChromaVectorStore:
    def __init__(self, embedding_size: int):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name="traditional_chunks",
            metadata={"hnsw:space": "cosine"}
        )
        self.chunks: List[Tuple[str, int, int, str]] = []
        self.duplicate_threshold = 0.95

    def add_embeddings(self, embeddings: np.ndarray, chunks: List[Tuple[str, int, int, str]]):
        if len(embeddings) == 0:
            return
            
        # Convert embeddings to list format for Chroma
        embeddings_list = embeddings.tolist()
        
        # Generate IDs for the chunks
        ids = [str(i + len(self.chunks)) for i in range(len(chunks))]
        
        # Add embeddings to Chroma
        self.collection.add(
            embeddings=embeddings_list,
            documents=[chunk[0] for chunk in chunks],
            ids=ids,
            metadatas=[{
                "doc_id": chunk[3],
                "char_start": chunk[1],
                "char_end": chunk[2]
            } for chunk in chunks]
        )
        
        # Store chunks for reference
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[SearchResult]:
        # Ensure query_embedding is 2D and convert to list
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search in Chroma
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=k
        )
        
        # Convert results to SearchResult format
        search_results = []
        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i]
            search_results.append(SearchResult(
                document_id=metadata["doc_id"],
                chunk_text=results['documents'][0][i],
                score=float(results['distances'][0][i]),
                char_start=metadata["char_start"],
                char_end=metadata["char_end"]
            ))
        
        return search_results

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

def compare_retrieval_quality(docs_directory: str = "documents"):
    # Create or open the output file
    with open('comparison_results.txt', 'w', encoding='utf-8') as f:
        f.write("RAG Comparison Results\n")
        f.write("=" * 80 + "\n\n")
        
        print("Initializing embedders...")
        f.write("Initializing embedders...\n")
        late_chunking_embedder = LateChunkingEmbedder(max_length=8000)
        traditional_embedder = TraditionalEmbedder()

        # Get sample embedding size
        sample_emb = traditional_embedder.embed_chunks(["test"])[0]
        embedding_size = len(sample_emb)

        # Initialize vector stores
        late_chunking_store = FaissVectorStore(late_chunking_embedder.embedding_size)
        traditional_store = ChromaVectorStore(traditional_embedder.embedding_size)

        # Load documents from directory
        try:
            documents = load_documents_from_directory(docs_directory)
            print(f"\nLoaded {len(documents)} documents from {docs_directory}")
            f.write(f"\nLoaded {len(documents)} documents from {docs_directory}\n")
        except Exception as e:
            print(f"Error loading documents: {str(e)}")
            return

        print("\nProcessing documents with late chunking approach...")
        f.write("\nProcessing documents with late chunking approach...\n")
        for doc in tqdm(documents):
            # Late chunking approach
            chunks_with_embeddings = late_chunking_embedder.embed_text(doc.text)
            embeddings = np.array([c.embedding for c in chunks_with_embeddings])
            chunks_info = [(c.text, c.start_char, c.end_char, doc.id) for c in chunks_with_embeddings]
            late_chunking_store.add_embeddings(embeddings, chunks_info)

        print("\nProcessing documents with traditional approach...")
        f.write("\nProcessing documents with traditional approach...\n")
        for doc in tqdm(documents):
            # Traditional approach
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
    # You can specify a different directory by passing it as an argument
    compare_retrieval_quality("documents")
