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
from typing import List, Tuple, Dict
import torch
from transformers import AutoTokenizer, AutoModel
import os
import warnings

# Suppress OpenMP warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Configure logging to be less verbose
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')

template_dir = Path(__file__).parent.parent / 'data' / 'templates'
app = Flask(__name__, template_folder=str(template_dir))
logger = logging.getLogger(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None

def load_model():
    """Load the embedding model and tokenizer"""
    global model, tokenizer
    if model is None:
        model_name = "jinaai/jina-embeddings-v2-base-en"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Use eager attention implementation to avoid warnings
        model = AutoModel.from_pretrained(
            model_name,
            attn_implementation="eager",
            trust_remote_code=True
        )
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()

def embed_query(query: str) -> np.ndarray:
    """Embed a query string using the loaded model"""
    load_model()
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    
    return embeddings[0]

def load_embedding_data(model_name=None):
    """Load embedding data from saved vector stores"""
    vector_store_dir = Path("outputs/comprehensive_comparison/vector_stores")
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

def find_closest_matches(query_embedding: np.ndarray, store_data: Dict, k: int = 10) -> Tuple[List[int], List[float]]:
    """Find k closest matches to the query embedding"""
    distances, indices = store_data['index'].search(query_embedding.reshape(1, -1), k)
    return indices[0].tolist(), distances[0].tolist()

def create_visualization_data(store_data, title, highlight_indices=None):
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
        'text': [chunk.text for chunk in store_data['chunks']],
        'is_match': [i in (highlight_indices or []) for i in range(len(store_data['chunks']))]
    })
    
    # First create figure with non-matching points
    fig = px.scatter(
        df[~df['is_match']],
        x='x',
        y='y',
        color='doc_id',
        title=title,
        hover_data=['text'],
        labels={'doc_id': 'Document'},
        template='plotly_white'
    )
    
    # Update base points to be smaller
    for trace in fig.data:
        trace.update(
            marker=dict(
                size=8,
            ),
            hoverinfo='text',
            hovertemplate='%{text}<extra></extra>',
        )

    # Choose your preferred symbol here
    match_symbol = 'diamond-cross'  # Other options: star, diamond, cross, x, pentagon, hexagon, asterisk, hash

    # Now add matching points in a separate layer
    if highlight_indices is not None and len(df[df['is_match']]) > 0:
        # Get the color mapping from the base figure
        color_map = {trace.name: trace.marker.color for trace in fig.data}
        
        # Create a new trace for matches with higher z-index
        for doc_id in df[df['is_match']]['doc_id'].unique():
            doc_matches = df[df['is_match'] & (df['doc_id'] == doc_id)]
            
            # Add an invisible larger point for better hover area
            hover_trace = {
                'type': 'scatter',
                'x': doc_matches['x'],
                'y': doc_matches['y'],
                'mode': 'markers',
                'marker': {
                    'size': 25,
                    'color': 'rgba(0,0,0,0)',
                },
                'name': f'{doc_id} (Match)',
                'text': doc_matches['text'],
                'hoverinfo': 'text',
                'showlegend': False,
                'hovertemplate': '%{text}<extra></extra>',
            }
            fig.add_trace(hover_trace)
            
            # Add the visible match point
            match_trace = {
                'type': 'scatter',
                'x': doc_matches['x'],
                'y': doc_matches['y'],
                'mode': 'markers',
                'marker': {
                    'size': 20,
                    'color': color_map[doc_id],
                    'line': {
                        'color': '#FFD700',
                        'width': 3
                    },
                    'symbol': match_symbol,
                    'opacity': 0.9
                },
                'name': f'{doc_id} (Match)',
                'text': doc_matches['text'],
                'hoverinfo': 'text',
                'showlegend': True,
                'hovertemplate': '%{text}<extra></extra>',
            }
            fig.add_trace(match_trace)
    
    # Update layout
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
            x=1.05,
            itemsizing='constant'
        ),
        margin=dict(r=200)
    )
    
    # Ensure matches are always on top by updating the layout
    fig.update_layout(
        yaxis=dict(
            range=[df['y'].min() - 0.5, df['y'].max() + 0.5],
            layer='above traces'
        ),
        xaxis=dict(
            range=[df['x'].min() - 0.5, df['x'].max() + 0.5],
            layer='above traces'
        )
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

@app.route('/process_query', methods=['POST'])
def process_query():
    """Process a query and return visualization with highlighted matches"""
    try:
        query = request.json.get('query', '')
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Load stores and embed query
        stores = load_embedding_data()
        query_embedding = embed_query(query)
        
        # Find closest matches for both stores
        late_indices, late_distances = find_closest_matches(query_embedding, stores['late_chunking'])
        trad_indices, trad_distances = find_closest_matches(query_embedding, stores['traditional'])
        
        # Create visualizations with highlighted matches
        late_chunking_viz = create_visualization_data(
            stores['late_chunking'],
            "Late Chunking Embeddings",
            highlight_indices=late_indices
        )
        
        traditional_viz = create_visualization_data(
            stores['traditional'],
            "Traditional Chunking Embeddings",
            highlight_indices=trad_indices
        )
        
        # Get matching chunks text
        late_matches = [
            {
                'text': stores['late_chunking']['chunks'][idx].text,
                'doc_id': stores['late_chunking']['chunks'][idx].doc_id,
                'distance': float(dist)
            }
            for idx, dist in zip(late_indices, late_distances)
        ]
        
        trad_matches = [
            {
                'text': stores['traditional']['chunks'][idx].text,
                'doc_id': stores['traditional']['chunks'][idx].doc_id,
                'distance': float(dist)
            }
            for idx, dist in zip(trad_indices, trad_distances)
        ]
        
        return jsonify({
            'late_chunking': late_chunking_viz,
            'traditional': traditional_viz,
            'matches': {
                'late_chunking': late_matches,
                'traditional': trad_matches
            }
        })
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
