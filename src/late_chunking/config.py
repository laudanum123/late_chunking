"""Configuration management using Pydantic."""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from pathlib import Path
import yaml

class EmbedderConfig(BaseModel):
    """Configuration for embedder models."""
    name: str = Field(..., description="Name or path of the embedding model")
    type: str = Field(..., description="Type of embedder (huggingface or openai)")
    embedding_size: int = Field(..., description="Size of output embeddings")
    max_length: int = Field(..., description="Maximum sequence length")
    api_key: Optional[str] = Field(None, description="Optional API key for cloud services")
    additional_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional model parameters"
    )

class Config(BaseModel):
    """Main configuration for the late chunking project."""
    default_embedder: str = Field(..., description="Default embedder to use")
    embedders: Dict[str, EmbedderConfig] = Field(
        ...,
        description="Configuration for different embedders"
    )
    output_dir: Path = Field(
        default=Path("output"),
        description="Directory for output files"
    )
    cache_dir: Optional[Path] = Field(
        default=None,
        description="Directory for caching"
    )
    
    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True

def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Config: Configuration object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    return Config(**config_dict)
