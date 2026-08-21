import argparse
import os
import json
import numpy as np
import chromadb
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.io as pio

class VectorVisualizer:
    """
    A class to handle retrieving data from ChromaDB and 
    generating a t-SNE visualization for the D&D RAG database.
    """
    
    def __init__(self):
        # Setup paths specifically for DandDCapstone
        from rag_config import DB_DIR, COLLECTION_NAME
        self.db_path = DB_DIR
        self.collection_name = COLLECTION_NAME
        
        print(f"Connecting to ChromaDB at: {self.db_path}")
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_collection(name=self.collection_name)
    
    def fetch_data(self):
        """Retrieves embeddings, text, and metadata from the store."""
        print(f"Fetching vectors from '{self.collection_name}'...")
        results = self.collection.get(include=['embeddings', 'documents', 'metadatas'])
        
        if len(results['embeddings']) == 0:
            raise ValueError("No embeddings found in the collection. Did you run the ingestion yet?")
            
        return results

    def generate_visualization(self, output_html="vector_viz.html", dimensions=3):
        """Performs t-SNE and saves an interactive Plotly HTML file in 2D or 3D."""
        data = self.fetch_data()
        
        vectors = np.array(data['embeddings'])
        documents = data['documents']
        metadatas = data['metadatas']
        
        # t-SNE to reduce to 2D/3D space
        print(f"Running t-SNE dimensionality reduction to {dimensions}D (this might take a moment)...")
        tsne = TSNE(n_components=dimensions, random_state=42, init='pca', learning_rate='auto')
        reduced_vectors = tsne.fit_transform(vectors)
        
        # Prepare labels based on the metadata schema we built
        hover_text = []
        colors = []
        for meta, doc in zip(metadatas, documents):
            doc_type = meta.get('type', 'text')
            
            # Color code based on text chunk vs table summary
            if doc_type == 'table':
                colors.append(1.0) # Highlight tables differently
                snippet = f"[TABLE SUMMARY] {doc[:100]}"
            else:
                colors.append(0.0) 
                snippet = doc[:100].replace('\n', ' ')
                
            hover_text.append(f"Type: {doc_type}<br>Snippet: {snippet}...")

        print("Creating interactive plot...")
        if dimensions == 3:
            trace = go.Scatter3d(
                x=reduced_vectors[:, 0],
                y=reduced_vectors[:, 1],
                z=reduced_vectors[:, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    opacity=0.7,
                    line=dict(width=0.5, color='DarkSlateGrey'),
                    color=colors,
                    colorscale='Jet',
                ),
                text=hover_text,
                hoverinfo='text'
            )
            layout_args = dict(
                scene=dict(
                    xaxis_title='Dimension 1',
                    yaxis_title='Dimension 2',
                    zaxis_title='Dimension 3'
                )
            )
        else:
            trace = go.Scatter(
                x=reduced_vectors[:, 0],
                y=reduced_vectors[:, 1],
                mode='markers',
                marker=dict(
                    size=7,
                    opacity=0.7,
                    line=dict(width=1, color='DarkSlateGrey'),
                    color=colors,
                    colorscale='Jet',
                ),
                text=hover_text,
                hoverinfo='text'
            )
            layout_args = dict(
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
            )

        fig = go.Figure(data=[trace])

        fig.update_layout(
            title=f"{dimensions}D Projection of {self.collection_name} (t-SNE)",
            template="plotly_dark",
            width=1000,
            height=700,
            **layout_args
        )

        # Outputting to HTML since we are in terminal
        pio.write_html(fig, file=output_html, auto_open=False)
        print(f"✅ Success! Visualization saved to: {os.path.abspath(output_html)}")


def main():
    parser = argparse.ArgumentParser(description="Generate 2D or 3D vector visualizations for D&D Rules.")
    parser.add_argument(
        "--dim", 
        type=int, 
        choices=[0, 2, 3], 
        default=0, 
        help="Dimension output: 2 for 2D, 3 for 3D, 0 for both (default: 0)"
    )
    args = parser.parse_args()

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        viz = VectorVisualizer()

        if args.dim in (0, 2):
            output_2d = os.path.join(script_dir, "dnd_rag_visualization_2d.html")
            viz.generate_visualization(output_2d, dimensions=2)
            
        if args.dim in (0, 3):
            output_3d = os.path.join(script_dir, "dnd_rag_visualization_3d.html")
            viz.generate_visualization(output_3d, dimensions=3)

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
