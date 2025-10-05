"""
TF-IDF based ranking algorithm
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from typing import List, Tuple, Dict
import config


class TFIDFRanker:
    """TF-IDF based document ranker with cosine similarity"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            min_df=config.TFIDF_MIN_DF,
            max_df=config.TFIDF_MAX_DF,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)
        )
        self.doc_vectors = None
        self.documents = []
        
    def index_documents(self, documents: List[Dict]) -> None:
        """
        Index documents for TF-IDF ranking
        
        Args:
            documents: List of document dictionaries with 'id', 'title', 'content'
        """
        self.documents = documents
        
        # Combine title and content for better representation
        texts = [
            f"{doc.get('title', '')} {doc.get('content', '')}" 
            for doc in documents
        ]
        
        # Fit and transform documents
        self.doc_vectors = self.vectorizer.fit_transform(texts)
        
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search and rank documents for a query
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of (doc_id, score) tuples sorted by relevance
        """
        if self.doc_vectors is None:
            raise ValueError("Documents not indexed. Call index_documents first.")
        
        # Transform query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return doc_id and score pairs
        results = [
            (self.documents[idx]['id'], float(similarities[idx]))
            for idx in top_indices
            if similarities[idx] > 0  # Filter out zero similarity
        ]
        
        return results
    
    def get_document_by_id(self, doc_id: str) -> Dict:
        """Get document by ID"""
        for doc in self.documents:
            if doc['id'] == doc_id:
                return doc
        return None
    
    def save_model(self, path: str = None) -> None:
        """Save the trained model"""
        if path is None:
            path = config.TFIDF_MODEL_PATH
        
        model_data = {
            'vectorizer': self.vectorizer,
            'doc_vectors': self.doc_vectors,
            'documents': self.documents
        }
        joblib.dump(model_data, path)
        
    def load_model(self, path: str = None) -> None:
        """Load a trained model"""
        if path is None:
            path = config.TFIDF_MODEL_PATH
        
        model_data = joblib.load(path)
        self.vectorizer = model_data['vectorizer']
        self.doc_vectors = model_data['doc_vectors']
        self.documents = model_data['documents']
