"""
Feature extraction for learning-to-rank
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import math
from typing import Dict, List
import config


class FeatureExtractor:
    """Extract ranking features from query-document pairs"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            lowercase=True
        )
        self.documents = []
        self.doc_vectors = None
        self.idf_scores = {}
        self.avgdl = 0
        
    def index_documents(self, documents: List[Dict]) -> None:
        """Index documents for feature extraction"""
        self.documents = documents
        
        # Build TF-IDF vectors
        texts = [f"{doc.get('title', '')} {doc.get('content', '')}" for doc in documents]
        self.doc_vectors = self.vectorizer.fit_transform(texts)
        
        # Calculate IDF scores
        feature_names = self.vectorizer.get_feature_names_out()
        idf_values = self.vectorizer.idf_
        self.idf_scores = dict(zip(feature_names, idf_values))
        
        # Calculate average document length
        doc_lengths = [len(text.split()) for text in texts]
        self.avgdl = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0
    
    def extract_features(self, query: str, document: Dict) -> np.ndarray:
        """
        Extract features for a query-document pair
        
        Args:
            query: Search query
            document: Document dictionary
            
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        # Text similarity features
        features.extend(self._text_similarity_features(query, document))
        
        # Query features
        features.extend(self._query_features(query))
        
        # Document features
        features.extend(self._document_features(document))
        
        # Term matching features
        features.extend(self._term_matching_features(query, document))
        
        # Statistical features
        features.extend(self._statistical_features(query, document))
        
        return np.array(features)
    
    def _text_similarity_features(self, query: str, document: Dict) -> List[float]:
        """Text similarity features"""
        doc_text = f"{document.get('title', '')} {document.get('content', '')}"
        
        # TF-IDF cosine similarity
        query_vec = self.vectorizer.transform([query])
        doc_idx = next((i for i, d in enumerate(self.documents) if d['id'] == document['id']), None)
        
        if doc_idx is not None and self.doc_vectors is not None:
            doc_vec = self.doc_vectors[doc_idx]
            tfidf_sim = float(cosine_similarity(query_vec, doc_vec)[0][0])
        else:
            tfidf_sim = 0.0
        
        # Title similarity (simple word overlap)
        title_sim = self._calculate_overlap(query, document.get('title', ''))
        
        # BM25 score
        bm25_score = self._calculate_bm25(query, doc_text)
        
        return [tfidf_sim, title_sim, bm25_score]
    
    def _query_features(self, query: str) -> List[float]:
        """Query-specific features"""
        query_tokens = query.lower().split()
        
        return [
            len(query_tokens),  # Query length
            len(set(query_tokens)),  # Unique terms
            len(query) / len(query_tokens) if query_tokens else 0  # Avg term length
        ]
    
    def _document_features(self, document: Dict) -> List[float]:
        """Document-specific features"""
        content = document.get('content', '')
        title = document.get('title', '')
        
        content_tokens = content.split()
        
        features = [
            len(content_tokens),  # Document length
            len(title.split()),  # Title length
            document.get('quality_score', 0.5),  # Document quality (0-1)
            document.get('pagerank', 0.5),  # Authority score
            document.get('freshness', 0.5),  # Content freshness (0-1)
        ]
        
        return features
    
    def _term_matching_features(self, query: str, document: Dict) -> List[float]:
        """Term matching and coverage features"""
        query_tokens = set(query.lower().split())
        doc_text = f"{document.get('title', '')} {document.get('content', '')}".lower()
        doc_tokens = set(doc_text.split())
        title_tokens = set(document.get('title', '').lower().split())
        
        # Query coverage
        matched_terms = query_tokens.intersection(doc_tokens)
        query_coverage = len(matched_terms) / len(query_tokens) if query_tokens else 0
        
        # Title matches
        title_matches = len(query_tokens.intersection(title_tokens))
        
        # Exact phrase match
        exact_match = 1.0 if query.lower() in doc_text else 0.0
        
        return [query_coverage, title_matches, exact_match]
    
    def _statistical_features(self, query: str, document: Dict) -> List[float]:
        """Statistical features"""
        doc_text = f"{document.get('title', '')} {document.get('content', '')}"
        query_tokens = query.lower().split()
        doc_tokens = doc_text.lower().split()
        
        # Term frequency statistics
        doc_term_freq = Counter(doc_tokens)
        
        max_tf = max([doc_term_freq.get(term, 0) for term in query_tokens]) if query_tokens else 0
        sum_tf = sum([doc_term_freq.get(term, 0) for term in query_tokens])
        
        # Position features (simplified)
        first_occurrence = self._first_occurrence_position(query_tokens, doc_tokens)
        
        return [max_tf, sum_tf, first_occurrence]
    
    def _calculate_overlap(self, text1: str, text2: str) -> float:
        """Calculate simple word overlap"""
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_bm25(self, query: str, document: str, k1: float = 1.5, b: float = 0.75) -> float:
        """Calculate BM25 score"""
        query_tokens = query.lower().split()
        doc_tokens = document.lower().split()
        doc_term_freq = Counter(doc_tokens)
        doc_len = len(doc_tokens)
        
        score = 0.0
        for term in query_tokens:
            tf = doc_term_freq.get(term, 0)
            idf = self.idf_scores.get(term, 0)
            
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / self.avgdl)) if self.avgdl > 0 else 1
            score += idf * (numerator / denominator)
        
        return score
    
    def _first_occurrence_position(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """Find normalized position of first query term in document"""
        if not doc_tokens or not query_tokens:
            return 1.0
        
        positions = []
        doc_tokens_lower = [t.lower() for t in doc_tokens]
        
        for term in query_tokens:
            try:
                pos = doc_tokens_lower.index(term.lower())
                positions.append(pos)
            except ValueError:
                continue
        
        if not positions:
            return 1.0
        
        # Return normalized position (0 = start, 1 = end)
        return min(positions) / len(doc_tokens)
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features"""
        return [
            # Text similarity features (3)
            'tfidf_similarity',
            'title_similarity',
            'bm25_score',
            
            # Query features (3)
            'query_length',
            'query_unique_terms',
            'query_avg_term_length',
            
            # Document features (5)
            'doc_length',
            'title_length',
            'quality_score',
            'pagerank',
            'freshness',
            
            # Term matching features (3)
            'query_coverage',
            'title_matches',
            'exact_match',
            
            # Statistical features (3)
            'max_term_frequency',
            'sum_term_frequency',
            'first_occurrence_position'
        ]
