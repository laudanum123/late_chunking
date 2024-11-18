from abc import ABC, abstractmethod
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List
import openai
import os
import yaml
from dataclasses import dataclass
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class EmbeddingConfig:
    name: str
    type: str
    embedding_size: int
    max_length: int
    api_key: str = None

class BaseEmbedder(ABC):
    @abstractmethod
    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        pass

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks using sentence boundaries"""
        # Simple sentence splitting using common endings
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

class HuggingFaceEmbedder(BaseEmbedder):
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(config.name, trust_remote_code=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.embedding_size = config.embedding_size

    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        embeddings = []
        for chunk in chunks:
            # Tokenize and truncate
            inputs = self.tokenizer(chunk, 
                                  return_tensors='pt', 
                                  max_length=self.config.max_length,
                                  truncation=True,
                                  padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token embedding
                embedding = outputs.last_hidden_state[:, 0].cpu().numpy()
                embeddings.append(embedding[0])
        
        return np.array(embeddings)

class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        # Try to get API key from config first, then environment
        api_key = config.api_key if config.api_key else os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in config or .env file")
        openai.api_key = api_key
        self.embedding_size = config.embedding_size

    def _chunk_text(self, text: str) -> List[str]:
        # Use the implementation from the base class
        return super()._chunk_text(text)

    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        """Embed a list of text chunks using OpenAI's API"""
        if not chunks:
            return np.array([], dtype=np.float32)

        embeddings = []
        for chunk in chunks:
            # Truncate text if needed (OpenAI has a token limit)
            if len(chunk) > self.config.max_length:
                chunk = chunk[:self.config.max_length]
            
            response = openai.Embedding.create(
                input=chunk,
                model=self.config.name
            )
            embedding = response['data'][0]['embedding']
            embeddings.append(embedding)
        
        return np.array(embeddings, dtype=np.float32)

def load_config(config_path: str = "config.yaml") -> dict:
    """Load and process configuration, interpolating environment variables"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle environment variable interpolation
    if 'embedding_models' in config:
        for model_config in config['embedding_models'].values():
            if 'api_key' in model_config and model_config['api_key'].startswith('${'):
                env_var = model_config['api_key'][2:-1]  # Remove ${ and }
                model_config['api_key'] = os.getenv(env_var)
    
    return config

def create_embedder(model_name: str = None, config_path: str = "config.yaml") -> BaseEmbedder:
    """Factory function to create embedder instances"""
    config = load_config(config_path)
    
    if model_name is None:
        model_name = config['default_model']
    
    if model_name not in config['embedding_models']:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_config = config['embedding_models'][model_name]
    embedding_config = EmbeddingConfig(**model_config)
    
    if model_config['type'] == 'huggingface':
        return HuggingFaceEmbedder(embedding_config)
    elif model_config['type'] == 'openai':
        return OpenAIEmbedder(embedding_config)
    else:
        raise ValueError(f"Unknown embedder type: {model_config['type']}")
