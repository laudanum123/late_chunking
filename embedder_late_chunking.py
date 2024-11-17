from transformers import AutoModel, AutoTokenizer
import numpy as np
import torch
from typing import List, Tuple, Union, Optional
from dataclasses import dataclass

@dataclass
class ChunkWithEmbedding:
    text: str
    embedding: np.ndarray
    start_char: int
    end_char: int

class LateChunkingEmbedder:
    def __init__(
        self, 
        model_name: str = 'jinaai/jina-embeddings-v2-base-en',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        max_length: Optional[int] = None
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(device)
        self.device = device
        self.max_length = max_length
        # Get embedding size from model config
        self.embedding_size = self.model.config.hidden_size

    def _chunk_by_sentences(self, input_text: str) -> Tuple[List[str], List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Split the input text into sentences and return chunks with their token and character spans
        """
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

    def _late_chunking(
        self, 
        model_output: torch.Tensor, 
        span_annotations: List[Tuple[int, int]]
    ) -> List[np.ndarray]:
        """
        Create embeddings using the late chunking approach
        """
        token_embeddings = model_output[0]
        outputs = []
        
        for embeddings, annotations in zip(token_embeddings, [span_annotations]):
            if self.max_length is not None:
                annotations = [
                    (start, min(end, self.max_length - 1))
                    for (start, end) in annotations
                    if start < (self.max_length - 1)
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
        """
        Split text into macro chunks that fit within model's maximum context size.
        Returns list of (chunk_text, start_position) tuples.
        """
        if not self.max_length:
            return [(text, 0)]
            
        # Get rough character limit based on max tokens
        # Using a conservative estimate of 4 characters per token
        char_limit = self.max_length * 4
        
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

    def embed_text(self, text: str) -> List[ChunkWithEmbedding]:
        """
        Embed text using late chunking approach, returning chunks with their embeddings
        and character positions. Handles texts exceeding maximum context size through
        macro-chunking.
        """
        # First split into macro chunks if needed
        macro_chunks = self._create_macro_chunks(text)
        all_results = []
        
        for chunk_text, chunk_start in macro_chunks:
            # Get chunks and their spans for this macro chunk
            chunks, token_spans, char_spans = self._chunk_by_sentences(chunk_text)
            
            # Get contextualized embeddings
            inputs = self.tokenizer(chunk_text, return_tensors='pt').to(self.device)
            with torch.no_grad():
                model_output = self.model(**inputs)
            
            # Create embeddings using late chunking
            embeddings = self._late_chunking(model_output, token_spans)
            
            # Combine chunks with their embeddings and spans
            # Adjust character positions based on macro chunk start position
            for chunk, embedding, (start_char, end_char) in zip(chunks, embeddings, char_spans):
                all_results.append(
                    ChunkWithEmbedding(
                        text=chunk,
                        embedding=embedding,
                        start_char=chunk_start + start_char,
                        end_char=chunk_start + end_char
                    )
                )
        
        return all_results

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query string for retrieval
        """
        inputs = self.tokenizer(query, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Average pool all token embeddings instead of using CLS token
            embeddings = outputs[0][0]
            mean_embedding = embeddings.mean(dim=0)
            # L2 normalize the embedding
            normalized_embedding = mean_embedding / torch.norm(mean_embedding)
            return normalized_embedding.cpu().numpy()

    @staticmethod
    def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        Assumes embeddings are already L2 normalized.
        """
        return np.dot(embedding1, embedding2)

# Example usage
if __name__ == "__main__":
    embedder = LateChunkingEmbedder()
    
    input_text = "Berlin is the capital and largest city of Germany, both by area and by population. Its more than 3.85 million inhabitants make it the European Union's most populous city, as measured by population within city limits. The city is also one of the states of Germany, and is the third smallest state in the country in terms of area."
    
    # Get chunks with embeddings
    chunks_with_embeddings = embedder.embed_text(input_text)
    
    # Example query
    query = "Berlin"
    query_embedding = embedder.embed_query(query)
    
    # Print chunks and their similarities with the query
    print(f"Query: {query}\n")
    print("Chunks and their similarities:")
    for chunk in chunks_with_embeddings:
        similarity = embedder.compute_similarity(query_embedding, chunk.embedding)
        print(f'Chunk: "{chunk.text}"')
        print(f'Similarity: {similarity:.4f}\n')