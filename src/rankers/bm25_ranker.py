"""
BM25 ranking algorithm implementation
"""
import numpy as np
import math
from collections import Counter
from typing import List, Tuple, Dict
import joblib
import config


class BM25Ranker:
    """BM25 probabilistic ranking algorithm"""
    
    def __init__(self, k1: float = None, b: float = None):
        """
        Initialize BM25 ranker
        
        Args:
            k1: Term frequency saturation parameter (default from config)
            b: Length normalization parameter (default from config)
        """
        self.k1 = k1 if k1 is not None else config.BM25_K1
        self.b = b if b is not None else config.BM25_B
        
        self.documents = []
        self.doc_freqs = {}
        self.idf = {}
        self.doc_lengths = []
        self.avgdl = 0
        self.N = 0
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization (can be enhanced with NLTK)"""
        return text.lower().split()
    
    def _calculate_idf(self) -> None:
        """Calculate IDF scores for all terms"""
        self.idf = {}
        for term, df in self.doc_freqs.items():
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
    
    def index_documents(self, documents: List[Dict]) -> None:
        """
        Index documents for BM25 ranking
        
        Args:
            documents: List of document dictionaries with 'id', 'title', 'content'
        """
        self.documents = documents
        self.N = len(documents)
        self.doc_freqs = {}
        self.doc_lengths = []
        
        # Calculate document frequencies and lengths
        for doc in documents:
            text = f"{doc.get('title', '')} {doc.get('content', '')}"
            tokens = self._tokenize(text)
            self.doc_lengths.append(len(tokens))
            
            # Count unique terms in document
            unique_terms = set(tokens)
            for term in unique_terms:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
        
        # Calculate average document length
        self.avgdl = sum(self.doc_lengths) / self.N if self.N > 0 else 0
        
        # Calculate IDF scores
        self._calculate_idf()
    
    def _score_document(self, query_terms: List[str], doc_idx: int) -> float:
        """
        Calculate BM25 score for a single document
        
        Args:
            query_terms: List of query terms
            doc_idx: Document index
            
        Returns:
            BM25 score
        """
        doc = self.documents[doc_idx]
        text = f"{doc.get('title', '')} {doc.get('content', '')}"
        doc_tokens = self._tokenize(text)
        doc_term_freqs = Counter(doc_tokens)
        doc_len = self.doc_lengths[doc_idx]
        
        score = 0.0
        for term in query_terms:
            if term not in self.idf:
                continue
            
            tf = doc_term_freqs.get(term, 0)
            idf = self.idf[term]
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
            score += idf * (numerator / denominator)
        
        return score
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search and rank documents for a query
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of (doc_id, score) tuples sorted by relevance
        """
        if not self.documents:
            raise ValueError("Documents not indexed. Call index_documents first.")
        
        query_terms = self._tokenize(query)
        
        # Score all documents
        scores = []
        for i in range(self.N):
            score = self._score_document(query_terms, i)
            if score > 0:
                scores.append((i, score))
        
        # Sort by score and get top k
        scores.sort(key=lambda x: x[1], reverse=True)
        top_scores = scores[:top_k]
        
        # Return doc_id and score pairs
        results = [
            (self.documents[idx]['id'], float(score))
            for idx, score in top_scores
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
            path = config.BM25_MODEL_PATH
        
        model_data = {
            'k1': self.k1,
            'b': self.b,
            'documents': self.documents,
            'doc_freqs': self.doc_freqs,
            'idf': self.idf,
            'doc_lengths': self.doc_lengths,
            'avgdl': self.avgdl,
            'N': self.N
        }
        joblib.dump(model_data, path)
    
    def load_model(self, path: str = None) -> None:
        """Load a trained model"""
        if path is None:
            path = config.BM25_MODEL_PATH
        
        model_data = joblib.load(path)
        self.k1 = model_data['k1']
        self.b = model_data['b']
        self.documents = model_data['documents']
        self.doc_freqs = model_data['doc_freqs']
        self.idf = model_data['idf']
        self.doc_lengths = model_data['doc_lengths']
        self.avgdl = model_data['avgdl']
        self.N = model_data['N']
