# Late Chunking RAG Comparison

A research project comparing traditional and late chunking approaches in Retrieval-Augmented Generation (RAG) systems.

## Overview

This project provides tools to compare different chunking strategies in RAG systems, with a focus on comparing traditional upfront chunking versus late chunking approaches. It includes visualization tools and comprehensive evaluation metrics to analyze the effectiveness of each approach.

## Features

- Advanced embedding strategies:
  - Traditional chunking with fixed-size chunks
  - Late chunking where every token has seen the whole context
  - Async-first design for better performance
- Visualization tools:
  - Interactive 2D visualization of embeddings using t-SNE
  - Web interface for exploring embedding spaces
- Comprehensive evaluation:
  - Similarity scores for each approach
  - Detailed chunk-by-chunk comparison
  - Document source tracking
- Built with modern Python:
  - Type-safe configuration using Pydantic
  - Async/await for better performance
  - FAISS for efficient similarity search
  - HuggingFace transformers for embeddings

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/late_chunking.git
cd late_chunking
```

2. Create and activate a virtual environment (Python 3.10+ required):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e ".[dev]"
```

## Usage

### Running a Comprehensive Comparison

1. Place your documents in the `documents/` directory.

2. Run the comprehensive comparison:
```bash
python examples/comprehensive_comparison.py
```

This will:
- Process all documents in the `documents/` directory
- Generate embeddings using both traditional and late chunking
- Save vector stores for visualization
- Output comparison results to `comparison_results.txt`

### Visualizing Embeddings

1. Start the web visualizer:
```bash
python web_visualizer.py
```

2. Open `http://localhost:5000` in your browser to see:
- Interactive 2D visualization of embeddings
- Color-coded document clusters
- Hover text showing chunk contents

### Programmatic Usage

```python
import asyncio
from late_chunking.rag_comparison import RAGComparison
from pathlib import Path

async def main():
    # Initialize comparison
    comparison = RAGComparison()
    
    # Load documents
    doc_dir = Path("documents")
    documents = []
    for doc_path in doc_dir.glob("*.txt"):
        with open(doc_path) as f:
            documents.append(f.read())
    
    # Define queries
    queries = [
        "What is the main topic of these documents?",
        "What are the key findings?"
    ]
    
    # Run comparison
    results = await comparison.run_comparison(queries, documents)
    
    # Print results
    for result in results:
        print(result)  # Shows scores, document sources, and chunks

if __name__ == "__main__":
    asyncio.run(main())
```

### Configuration

The system uses a default configuration optimized for general use. To customize:

1. Create a `config.yaml` in the project root:
```yaml
embedding_models:
  default:
    name: "jinaai/jina-embeddings-v2-base-en"
    type: "huggingface"
    embedding_size: 768
    max_length: 8192
    chunk_size: 512
    chunk_overlap: 50
```

2. Adjust parameters:
- `chunk_size`: Size of chunks for traditional chunking
- `chunk_overlap`: Overlap between chunks
- `max_length`: Maximum sequence length for the model

## Output Format

The comparison results (`comparison_results.txt`) show:
1. Query information
2. Traditional chunking results:
   - Similarity scores
   - Document sources
   - Chunk contents
3. Late chunking results:
   - Similarity scores
   - Document sources
   - Dynamically sized chunks

## Development

1. Install development dependencies:
```bash
pip install -e ".[dev]"
```

2. Set up pre-commit hooks:
```bash
pre-commit install
```

3. Run tests:
```bash
pytest
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - see LICENSE file for details