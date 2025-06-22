from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import numpy as np
import torch

class ContextAnalyzer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the ContextAnalyzer.
        Detects available hardware (GPU/CPU) and loads a pre-trained model.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading context analysis model '{model_name}' onto '{self.device}' device.")
        
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading SentenceTransformer model: {e}")
            print("Please ensure you have an internet connection and the required libraries are installed.")
            self.model = None

    def find_outliers(self, paragraphs: list[str], eps=0.7, min_samples=2) -> list[int]:
        """
        Analyzes a list of paragraphs to find semantic outliers.

        Args:
            paragraphs (list[str]): A list of text paragraphs.
            eps (float): The maximum distance for DBSCAN clustering. This value is
                         sensitive to the embedding model and the nature of the text.
            min_samples (int): The minimum number of samples for a DBSCAN cluster.

        Returns:
            list[int]: A list of indices corresponding to the outlier paragraphs.
        """
        if not self.model:
            print("Context analysis model not loaded. Skipping outlier detection.")
            return []
            
        if len(paragraphs) < min_samples * 2: # Need enough data to find a "main" cluster
            print("Not enough text to perform meaningful context analysis.")
            return []

        print("Generating text embeddings for context analysis...")
        embeddings = self.model.encode(paragraphs, show_progress_bar=True, normalize_embeddings=True)

        print("Clustering paragraphs to find outliers...")
        # Using euclidean distance on normalized vectors is equivalent to cosine distance.
        db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(embeddings)
        
        labels = db.labels_
        
        # Find the largest cluster, which we assume is the main context.
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        
        if len(counts) > 0:
            # Assume the largest cluster is the main topic
            largest_cluster_label = unique_labels[np.argmax(counts)]
            # Flag everything that is NOT in the largest cluster as an outlier
            outlier_indices = np.where(labels != largest_cluster_label)[0].tolist()
            print(f"Identified {len(outlier_indices)} potential outliers out of {len(paragraphs)} paragraphs.")
        else:
            # This can happen if DBSCAN labels all points as noise (-1)
            print("Could not identify a main context cluster. All paragraphs considered outliers.")
            outlier_indices = list(range(len(paragraphs)))

        return outlier_indices 