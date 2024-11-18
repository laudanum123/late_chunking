from flask import Flask, render_template, jsonify, request
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px
import json
import pandas as pd
from pathlib import Path
import faiss
import pickle
import logging

from late_chunking.embedders import EmbeddingConfig
from late_chunking.rag_comparison import RAGComparison

app = Flask(__name__)
logger = logging.getLogger(__name__)

def load_embedding_data(model_name=None):
    """Load embedding data from saved vector stores"""
    vector_store_dir = Path("vector_stores") / "comprehensive_comparison"
    late_store_path = vector_store_dir / "late_chunking"
    trad_store_path = vector_store_dir / "traditional"
    
    # Load FAISS indices and chunks
    try:
        # Load traditional store
        trad_index = faiss.read_index(str(trad_store_path / "index.faiss"))
        with open(trad_store_path / "chunks.pkl", 'rb') as f:
            trad_chunks = pickle.load(f)
            
        # Load late chunking store
        late_index = faiss.read_index(str(late_store_path / "index.faiss"))
        with open(late_store_path / "chunks.pkl", 'rb') as f:
            late_chunks = pickle.load(f)
            
        return {
            'traditional': {'index': trad_index, 'chunks': trad_chunks},
            'late_chunking': {'index': late_index, 'chunks': late_chunks}
        }
    except Exception as e:
        raise ValueError(f"Error loading vector stores: {str(e)}. Please run comprehensive_comparison.py first.")

def create_visualization_data(store_data, title):
    """Create Plotly visualization data from a vector store"""
    # Get all embeddings from the FAISS index
    num_vectors = store_data['index'].ntotal
    embeddings = np.zeros((num_vectors, store_data['index'].d))
    for i in range(num_vectors):
        embeddings[i] = store_data['index'].reconstruct(i)
    
    # Reduce dimensionality with t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    # Create DataFrame for Plotly
    df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'doc_id': [chunk.doc_id for chunk in store_data['chunks']],
        'text': [chunk.text for chunk in store_data['chunks']]
    })
    
    # Create Plotly figure
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='doc_id',
        title=title,
        hover_data=['text'],
        labels={'doc_id': 'Document'},
        template='plotly_white'
    )
    
    # Update layout for better visualization
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        ),
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        ),
        margin=dict(r=200)  # Add right margin for legend
    )
    
    # Update trace settings
    for trace in fig.data:
        trace.update(
            hoverinfo='none',  # Disable default hover
            hovertemplate=None
        )
    
    return json.loads(fig.to_json())

@app.route('/')
def index():
    return render_template('visualizer.html')

@app.route('/get_visualizations')
def get_visualizations():
    try:
        stores = load_embedding_data()
        
        late_chunking_viz = create_visualization_data(
            stores['late_chunking'], 
            "Late Chunking Embeddings"
        )
        
        traditional_viz = create_visualization_data(
            stores['traditional'],
            "Traditional Chunking Embeddings"
        )
        
        return jsonify({
            'late_chunking': late_chunking_viz,
            'traditional': traditional_viz
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
