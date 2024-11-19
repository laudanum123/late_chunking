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

1. Place your documents in the `src/late_chunking/data/documents/` directory.

2. Run the comprehensive comparison:
```bash
python examples/comprehensive_comparison.py
```

This will:
- Process all documents in the `src/late_chunking/data/documents/` directory
- Generate embeddings using both traditional and late chunking
- Save vector stores for visualization
- Output comparison results to `comparison_results.txt`

### Visualizing Embeddings

1. Start the web visualizer:
```bash
python -m src.late_chunking.visualization.web_visualizer
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
    doc_dir = Path("src/late_chunking/data/documents")
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

## Project Structure

```
late_chunking/
├── src/late_chunking/          # Main package directory
│   ├── data/                   # Data storage
│   │   ├── documents/         # Input documents for processing
│   │   └── templates/         # HTML and other templates
│   ├── embedders/             # Embedding implementations
│   │   ├── huggingface.py    # HuggingFace-based embedder
│   │   ├── late_chunking.py  # Late chunking embedder
│   │   └── openai.py         # OpenAI-based embedder
│   ├── utils/                 # Utility functions
│   │   └── wiki_extractor.py # Wikipedia content extraction
│   ├── vector_store/         # Vector storage implementations
│   │   └── stores/          # Generated vector stores
│   └── visualization/        # Visualization tools
│       ├── output/          # Generated visualizations
│       └── web_visualizer.py # Interactive web visualization
├── examples/                  # Example scripts
│   ├── basic_comparison.py   # Simple RAG comparison
│   └── comprehensive_comparison.py # Detailed analysis
├── tests/                    # Test suite
└── README.md                 # This file
```

### Key Components

1. **Data Management** (`src/late_chunking/data/`)
   - Place input documents in `documents/`
   - HTML templates for visualization in `templates/`

2. **Embedders** (`src/late_chunking/embedders/`)
   - Multiple embedding strategies
   - Configurable model parameters
   - Vector store integration

3. **Utilities** (`src/late_chunking/utils/`)
   - `wiki_extractor.py`: Extract content from Wikipedia
   - Add custom utilities for data processing

4. **Vector Stores** (`src/late_chunking/vector_store/`)
   - Generated embeddings stored in `stores/`
   - Organized by comparison type and strategy

5. **Visualization** (`src/late_chunking/visualization/`)
   - Generated plots saved to `output/`
   - Interactive web interface in `web_visualizer.py`

### Output Structure

Each example stores its outputs in a dedicated directory under `outputs/`:

```
outputs/
├── comprehensive_comparison/    # Outputs from comprehensive comparison
│   ├── logs/                  # Log files
│   │   └── comparison.log    # Detailed run logs
│   ├── vector_stores/        # Generated vector stores
│   │   ├── late_chunking/   # Late chunking embeddings
│   │   └── traditional/     # Traditional chunking embeddings
│   ├── visualizations/       # Generated visualizations
│   │   ├── late_embeddings.png
│   │   └── traditional_embeddings.png
│   └── results.txt           # Detailed comparison results
│
└── basic_comparison/          # Outputs from basic comparison
    ├── logs/                 # Log files
    ├── vector_stores/        # Vector stores
    └── results.txt          # Basic comparison results
```

### Output Types

1. **Vector Stores**
   - Location: `outputs/<example>/vector_stores/`
   - Contains FAISS indices and metadata
   - Separate stores for each chunking strategy

2. **Visualizations**
   - Location: `outputs/<example>/visualizations/`
   - Contains:
     * Embedding space visualizations (*.png)
     * Comparison plots
     * Analysis charts

3. **Logs**
   - Location: `outputs/<example>/logs/`
   - Contains:
     * Run logs (comparison.log)
     * Error logs
     * Performance metrics

4. **Results**
   - Location: `outputs/<example>/results.txt`
   - Contains:
     * Comparison results
     * Metrics and scores
     * Analysis summaries

## Development

### Setting Up Development Environment

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
pytest tests/
```

### Adding New Features

1. **New Embedder**
   - Add new embedder in `src/late_chunking/embedders/`
   - Inherit from `BaseEmbedder`
   - Update `__init__.py` to expose new class

2. **New Chunking Strategy**
   - Add strategy in `src/late_chunking/chunkers.py`
   - Implement required interfaces
   - Add tests in `tests/test_chunkers.py`

3. **New Visualization**
   - Add visualization code in `src/late_chunking/visualization/`
   - Update web interface if needed
   - Store outputs in `visualization/output/`

### Code Organization

- Keep embedders modular and independent
- Store all data files under `src/late_chunking/data/`
- Save all outputs to appropriate directories under `src/late_chunking/`
- Add examples to `examples/` directory

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - see LICENSE file for details