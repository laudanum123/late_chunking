from flask import Flask, render_template, jsonify, request
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px
import json
import faiss
import pickle
from pathlib import Path
import pandas as pd
import logging
from typing import Dict, List, Tuple
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
    if model is None or tokenizer is None:
        try:
            logger.info("Loading model and tokenizer...")
            model_name = "jinaai/jina-embeddings-v2-base-de"  # Changed to German model to match fixed_token_comparison.py
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
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            raise

def embed_query(query: str) -> np.ndarray:
    """Embed a query string using the loaded model"""
    global model, tokenizer
    load_model()
    try:
        logger.info(f"Embedding query: {query}")
        inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        logger.info("Query embedded successfully")
        return embeddings[0]
    except Exception as e:
        logger.error(f"Error embedding query: {str(e)}", exc_info=True)
        raise

def load_embedding_data(experiment=None):
    """Load vector stores from disk."""
    try:
        # Use absolute paths
        base_dir = Path(__file__).resolve().parent.parent.parent.parent / "outputs"
        if experiment == 'fixed':
            store_dir = base_dir / "fixed_token_comparison/vector_stores"
        else:  # default to comprehensive
            store_dir = base_dir / "comprehensive_comparison/vector_stores"
        
        late_store_path = store_dir / "late_chunking"
        trad_store_path = store_dir / "traditional"
        
        logger.info(f"Loading vector stores from {store_dir}")
        logger.info(f"Late chunking store path: {late_store_path}")
        logger.info(f"Traditional store path: {trad_store_path}")
            
        # Load FAISS indices
        late_index = faiss.read_index(str(late_store_path / "index.faiss"))
        trad_index = faiss.read_index(str(trad_store_path / "index.faiss"))
        logger.info(f"Loaded indices - Late: {late_index.ntotal} vectors, Traditional: {trad_index.ntotal} vectors")
        
        # Load chunks
        with open(late_store_path / "chunks.pkl", "rb") as f:
            late_data = pickle.load(f)
            late_chunks = late_data['chunks']
        with open(trad_store_path / "chunks.pkl", "rb") as f:
            trad_data = pickle.load(f)
            trad_chunks = trad_data['chunks']
        logger.info(f"Loaded chunks - Late: {len(late_chunks)} chunks, Traditional: {len(trad_chunks)} chunks")
            
        # Verify index and chunk counts match
        if late_index.ntotal != len(late_chunks):
            raise ValueError(f"Late chunking index ({late_index.ntotal} vectors) and chunks ({len(late_chunks)} chunks) mismatch")
        if trad_index.ntotal != len(trad_chunks):
            raise ValueError(f"Traditional index ({trad_index.ntotal} vectors) and chunks ({len(trad_chunks)} chunks) mismatch")
            
        # Extract embeddings from indices
        def get_embeddings_from_index(index):
            embeddings = np.zeros((index.ntotal, index.d), dtype=np.float32)
            for i in range(index.ntotal):
                try:
                    index.reconstruct(i, embeddings[i])
                except RuntimeError as e:
                    logger.error(f"Failed to reconstruct embedding {i}: {str(e)}")
                    embeddings[i] = np.zeros(index.d, dtype=np.float32)
            return embeddings
            
        return {
            'traditional': {
                'index': trad_index,
                'embeddings': get_embeddings_from_index(trad_index),
                'chunks': trad_chunks
            },
            'late_chunking': {
                'index': late_index,
                'embeddings': get_embeddings_from_index(late_index),
                'chunks': late_chunks
            }
        }
    except FileNotFoundError as e:
        logger.error(f"Vector store files not found: {str(e)}")
        raise ValueError(f"Vector stores not found in {store_dir}. Please run the comparison script first.")
    except Exception as e:
        logger.error(f"Error loading vector stores: {str(e)}", exc_info=True)
        raise ValueError(f"Error loading vector stores: {str(e)}. Please run the comparison script first.")

def find_closest_matches(query_embedding, store, k=10):
    """Find closest matches to query embedding in store."""
    if not store or 'embeddings' not in store:
        return [], []
    
    # Ensure query embedding is in correct shape
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    # Calculate cosine similarity with all embeddings
    query_norm = np.linalg.norm(query_embedding)
    scores = []
    indices = []
    
    # Get embeddings array from store
    embeddings = store['embeddings']
    for i, chunk_embedding in enumerate(embeddings):
        chunk_norm = np.linalg.norm(chunk_embedding)
        if query_norm == 0 or chunk_norm == 0:
            score = 0.0
        else:
            score = float(np.dot(query_embedding.flatten(), chunk_embedding) / (query_norm * chunk_norm))
        scores.append(score)
        indices.append(i)
    
    # Sort by score
    if not scores:
        return [], []
    sorted_pairs = sorted(zip(indices, scores), key=lambda x: x[1], reverse=True)
    top_indices, top_scores = zip(*sorted_pairs[:k])
    
    return list(top_indices), list(top_scores)

def get_chunk_text(chunk):
    """Extract text from chunk, handling both dict and object formats."""
    try:
        if isinstance(chunk, dict):
            text = chunk.get('text', 'unknown')
        else:
            text = chunk.text if hasattr(chunk, 'text') else str(chunk)
        # Ensure text is properly encoded
        return text.encode('utf-8', errors='replace').decode('utf-8')
    except Exception as e:
        logger.error(f"Error getting chunk text: {str(e)}")
        return 'unknown'

def get_chunk_doc_id(chunk):
    """Extract doc_id from chunk, handling both dict and object formats."""
    try:
        if isinstance(chunk, dict):
            doc_id = chunk.get('doc_id', 'unknown')
        else:
            doc_id = chunk.doc_id if hasattr(chunk, 'doc_id') else 'unknown'
        # Ensure doc_id is properly encoded
        return doc_id.encode('utf-8', errors='replace').decode('utf-8')
    except Exception as e:
        logger.error(f"Error getting chunk doc_id: {str(e)}")
        return 'unknown'

def get_chunk_id(chunk):
    """Extract id from chunk, handling both dict and object formats."""
    try:
        if isinstance(chunk, dict):
            chunk_id = chunk.get('id', 'unknown')
        else:
            chunk_id = chunk.id if hasattr(chunk, 'id') else 'unknown'
        # Ensure id is properly encoded
        return str(chunk_id).encode('utf-8', errors='replace').decode('utf-8')
    except Exception as e:
        logger.error(f"Error getting chunk id: {str(e)}")
        return 'unknown'

def get_chunk_parent_id(chunk):
    """Extract parent_id from chunk, handling both dict and object formats."""
    try:
        if isinstance(chunk, dict):
            parent_id = chunk.get('parent_id', 'unknown')
        else:
            parent_id = chunk.parent_id if hasattr(chunk, 'parent_id') else 'unknown'
        # Ensure parent_id is properly encoded
        return str(parent_id).encode('utf-8', errors='replace').decode('utf-8')
    except Exception as e:
        logger.error(f"Error getting chunk parent_id: {str(e)}")
        return 'unknown'

def create_visualization_data(store_data, title, highlight_indices=None):
    """Create Plotly visualization data from a vector store.
    
    Args:
        store_data (dict): Dictionary containing 'embeddings' and 'chunks'
        title (str): Title for the visualization
        highlight_indices (list): List of indices to highlight
    
    Returns:
        dict: Plotly figure data and layout
    """
    try:
        if not store_data or 'embeddings' not in store_data or 'chunks' not in store_data:
            logger.error("Invalid store data provided")
            return {'data': [], 'layout': {'title': title}}
        
        embeddings = store_data['embeddings']
        chunks = store_data['chunks']
        
        if len(embeddings) == 0 or len(chunks) == 0:
            logger.warning("Empty embeddings or chunks")
            return {'data': [], 'layout': {'title': title}}
            
        # Convert highlight_indices to list if it's a numpy array
        if highlight_indices is not None:
            if isinstance(highlight_indices, np.ndarray):
                highlight_indices = highlight_indices.tolist()
            elif not isinstance(highlight_indices, list):
                highlight_indices = list(highlight_indices)
        
        # Handle NaN values in embeddings
        embeddings = np.nan_to_num(embeddings, nan=0.0)
        
        # Create t-SNE projection with adjusted parameters
        n_samples = len(embeddings)
        perplexity = min(30, max(5, n_samples - 1))  # Adjust perplexity based on sample size
        
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=perplexity,
            init='pca',  # Use PCA initialization for better stability
            max_iter=1000,  # Updated from n_iter to max_iter
            learning_rate='auto'
        )
        
        try:
            projections = tsne.fit_transform(embeddings)
            
            # Handle any remaining NaN values in projections
            projections = np.nan_to_num(projections, nan=0.0)
            
            # Normalize projections to prevent extreme values
            for i in range(projections.shape[1]):
                if np.std(projections[:, i]) > 0:
                    projections[:, i] = (projections[:, i] - np.mean(projections[:, i])) / np.std(projections[:, i])
        
        except Exception as e:
            logger.error(f"t-SNE projection failed: {str(e)}")
            # Fallback to simple PCA if t-SNE fails
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42)
            projections = pca.fit_transform(embeddings)
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'x': projections[:, 0],
            'y': projections[:, 1],
            'text': [get_chunk_text(chunk) for chunk in chunks],
            'doc_id': [get_chunk_doc_id(chunk) for chunk in chunks],
            'id': [get_chunk_id(chunk) for chunk in chunks],
            'parent_id': [get_chunk_parent_id(chunk) for chunk in chunks],
            'is_match': [i in (highlight_indices or []) for i in range(len(chunks))]
        })
        
        # Get unique document IDs and create color map
        unique_docs = sorted(df['doc_id'].unique())  # Sort for consistent colors
        colors = px.colors.qualitative.Set3[:len(unique_docs)]  # Use Set3 color palette
        color_map = {doc: color for doc, color in zip(unique_docs, colors)}
        
        # Create a figure with a consistent color scheme
        fig = px.scatter(
            df,  # Use all points
            x='x',
            y='y',
            color='doc_id',
            title=title,
            hover_data=['text', 'id', 'parent_id'],
            labels={'doc_id': 'Document'},
            template='plotly_white',
            color_discrete_map=color_map  # Use our custom color map
        )
        
        # Update non-matching points to be smaller and more transparent
        fig.update_traces(
            selector=dict(mode='markers'),
            marker=dict(size=8, opacity=0.5),
            showlegend=True
        )
        
        # Add matching points with special styling
        df_matches = df[df['is_match']]
        if not df_matches.empty:
            match_symbol = 'star'
            
            # Group matches by document
            for doc_id in df_matches['doc_id'].unique():
                doc_matches = df_matches[df_matches['doc_id'] == doc_id]
                
                # Add an invisible larger point for better hover area
                hover_trace = {
                    'type': 'scatter',
                    'x': doc_matches['x'].tolist(),
                    'y': doc_matches['y'].tolist(),
                    'mode': 'markers',
                    'marker': {
                        'size': 25,
                        'color': 'rgba(0,0,0,0)',
                    },
                    'name': f'{doc_id} (Match)',
                    'text': doc_matches['text'].tolist(),
                    'hoverinfo': 'text',
                    'showlegend': False,
                    'hovertemplate': '%{text}<extra></extra>',
                }
                fig.add_trace(hover_trace)
                
                # Add the visible match point with consistent color
                match_trace = {
                    'type': 'scatter',
                    'x': doc_matches['x'].tolist(),
                    'y': doc_matches['y'].tolist(),
                    'mode': 'markers',
                    'marker': {
                        'size': 20,
                        'color': color_map.get(doc_id, '#808080'),  # Use gray as fallback color
                        'line': {
                            'width': 2,
                            'color': 'white'
                        },
                        'symbol': match_symbol,
                        'opacity': 0.9
                    },
                    'name': f'{doc_id} (Match)',
                    'text': doc_matches['text'].tolist(),
                    'hoverinfo': 'text',
                    'showlegend': True,
                    'hovertemplate': '%{text}<extra></extra>',
                }
                fig.add_trace(match_trace)
        
        # Update layout for better visualization
        fig.update_layout(
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255, 255, 255, 0.8)'
            ),
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor='white',
            title_x=0.5,
            title_y=0.95
        )
        
        # Update axes to be cleaner
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        return json.loads(fig.to_json())
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}", exc_info=True)
        raise

@app.route('/')
def index():
    return render_template('visualizer.html')

@app.route('/get_visualizations')
def get_visualizations():
    try:
        experiment = request.args.get('experiment', 'comprehensive')
        skip_viz = request.args.get('skip_viz', 'false').lower() == 'true'
        stores = load_embedding_data(experiment if experiment == 'fixed' else None)
        
        title_prefix = "Fixed Token" if experiment == 'fixed' else "Comprehensive"
        
        if skip_viz:
            # Return minimal data without computing TSNE
            return jsonify({
                'late_chunking': {
                    'data': [],
                    'layout': {'title': f"{title_prefix} Late Chunking Embeddings"}
                },
                'traditional': {
                    'data': [],
                    'layout': {'title': f"{title_prefix} Traditional Embeddings"}
                }
            })
        
        late_chunking_viz = create_visualization_data(
            stores['late_chunking'], 
            f"{title_prefix} Late Chunking Embeddings"
        )
        
        traditional_viz = create_visualization_data(
            stores['traditional'],
            f"{title_prefix} Traditional Embeddings"
        )
        
        return jsonify({
            'late_chunking': late_chunking_viz,
            'traditional': traditional_viz
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process_query', methods=['POST'])
def process_query():
    """Process a query and return visualization with highlighted matches."""
    try:
        data = request.get_json()
        query = data.get('query', '')
        experiment = data.get('experiment', 'fixed')
        k = data.get('k', 10)
        skip_viz = data.get('skip_viz', False)
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
            
        # Load embeddings
        stores = load_embedding_data(experiment)
        if not stores:
            return jsonify({'error': 'No embeddings found'}), 404
            
        # Embed query
        query_embedding = embed_query(query)
        if query_embedding is None:
            return jsonify({'error': 'Failed to embed query'}), 500
            
        # Find closest matches in both stores
        trad_indices, trad_distances = find_closest_matches(
            query_embedding,
            stores['traditional'],
            k=k
        )
        late_indices, late_distances = find_closest_matches(
            query_embedding,
            stores['late_chunking'],
            k=k
        )
        
        # Create response with matches
        response = {
            'matches': {
                'traditional': [
                    {
                        'text': get_chunk_text(stores['traditional']['chunks'][idx]),
                        'doc_id': get_chunk_doc_id(stores['traditional']['chunks'][idx]),
                        'distance': float(dist)
                    }
                    for idx, dist in zip(trad_indices, trad_distances)
                ],
                'late_chunking': [
                    {
                        'text': get_chunk_text(stores['late_chunking']['chunks'][idx]),
                        'doc_id': get_chunk_doc_id(stores['late_chunking']['chunks'][idx]),
                        'distance': float(dist)
                    }
                    for idx, dist in zip(late_indices, late_distances)
                ]
            }
        }
        
        # Create visualizations only if not skipped
        if not skip_viz:
            response['visualizations'] = {
                'traditional': create_visualization_data(
                    stores['traditional'],
                    'Traditional Chunking - Query Results',
                    highlight_indices=trad_indices
                ),
                'late_chunking': create_visualization_data(
                    stores['late_chunking'],
                    'Late Chunking - Query Results',
                    highlight_indices=late_indices
                )
            }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
