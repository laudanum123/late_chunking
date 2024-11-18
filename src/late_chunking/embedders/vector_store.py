"""Vector store functionality for embedders."""
from dataclasses import dataclass
import numpy as np
from typing import Optional, Dict, Any

@dataclass
class ChunkWithEmbedding:
    """A text chunk with its embedding."""
    text: str
    embedding: np.ndarray
    doc_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
