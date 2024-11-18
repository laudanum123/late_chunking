from flask import Flask, render_template, jsonify, request
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.utils
import json
import pandas as pd
from rag_comparison import FaissVectorStore
from embedders import load_config

app = Flask(__name__)

def load_embedding_data(model_name=None):
    """Load embedding data from saved vector stores"""
    store_suffix = f"_{model_name}" if model_name else ""
    late_chunking_store = FaissVectorStore.load(f"vector_stores/late_chunking{store_suffix}")
    traditional_store = FaissVectorStore.load(f"vector_stores/traditional{store_suffix}")
    
    if late_chunking_store is None or traditional_store is None:
        raise ValueError(f"Vector stores not found for model {model_name}. Please run rag_comparison.py first.")
    
    return late_chunking_store, traditional_store

def create_visualization_data(store, title):
    """Create Plotly visualization data from a vector store"""
    # Get all embeddings from the FAISS index
    num_vectors = store.index.ntotal
    embeddings = np.zeros((num_vectors, store.embedding_size))
    for i in range(num_vectors):
        embeddings[i] = store.index.reconstruct(i)
    
    # Reduce dimensionality with t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    # Create DataFrame for Plotly
    df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'doc_id': [chunk[3] for chunk in store.chunks],
        'text': [chunk[0] for chunk in store.chunks]
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
    # Load available models from config
    config = load_config()
    models = list(config['embedding_models'].keys())
    return render_template('visualizer.html', models=models)

@app.route('/get_visualizations')
def get_visualizations():
    try:
        model = request.args.get('model', None)
        late_chunking_store, traditional_store = load_embedding_data(model)
        
        late_chunking_viz = create_visualization_data(
            late_chunking_store, 
            f"Late Chunking Embeddings ({model if model else 'default'})"
        )
        
        traditional_viz = create_visualization_data(
            traditional_store,
            f"Traditional Chunking Embeddings ({model if model else 'default'})"
        )
        
        return jsonify({
            'late_chunking': late_chunking_viz,
            'traditional': traditional_viz
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
