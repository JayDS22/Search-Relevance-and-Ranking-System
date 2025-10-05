"""
LambdaMART learning-to-rank implementation
"""
import numpy as np
from xgboost import XGBRanker
from typing import List, Tuple, Dict
import joblib
import config
from src.features.feature_extractor import FeatureExtractor


class LambdaMARTRanker:
    """LambdaMART learning-to-rank algorithm using XGBoost"""
    
    def __init__(self):
        """Initialize LambdaMART ranker"""
        self.model = XGBRanker(
            objective='rank:ndcg',
            n_estimators=config.LAMBDAMART_N_ESTIMATORS,
            learning_rate=config.LAMBDAMART_LEARNING_RATE,
            max_depth=config.LAMBDAMART_MAX_DEPTH,
            min_child_weight=config.LAMBDAMART_MIN_CHILD_WEIGHT,
            subsample=config.LAMBDAMART_SUBSAMPLE,
            colsample_bytree=config.LAMBDAMART_COLSAMPLE_BYTREE,
            tree_method='hist'
        )
        
        self.feature_extractor = FeatureExtractor()
        self.documents = []
        self.is_trained = False
        
    def index_documents(self, documents: List[Dict]) -> None:
        """
        Index documents
        
        Args:
            documents: List of document dictionaries
        """
        self.documents = documents
        self.feature_extractor.index_documents(documents)
    
    def train(self, training_data: List[Dict]) -> None:
        """
        Train the LambdaMART model
        
        Args:
            training_data: List of training examples with format:
                {
                    'query': str,
                    'documents': List[str],  # doc_ids
                    'relevance_scores': List[int]  # 0-4 relevance labels
                }
        """
        if not self.documents:
            raise ValueError("Documents not indexed. Call index_documents first.")
        
        X_train = []
        y_train = []
        group_sizes = []
        
        for item in training_data:
            query = item['query']
            doc_ids = item['documents']
            relevance_scores = item['relevance_scores']
            
            query_features = []
            query_labels = []
            
            for doc_id, relevance in zip(doc_ids, relevance_scores):
                doc = self._get_document_by_id(doc_id)
                if doc:
                    features = self.feature_extractor.extract_features(query, doc)
                    query_features.append(features)
                    query_labels.append(relevance)
            
            if query_features:
                X_train.extend(query_features)
                y_train.extend(query_labels)
                group_sizes.append(len(query_features))
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Train the model
        self.model.fit(
            X_train, 
            y_train, 
            group=group_sizes,
            verbose=True
        )
        
        self.is_trained = True
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search and rank documents for a query
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of (doc_id, score) tuples sorted by relevance
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first or load a trained model.")
        
        if not self.documents:
            raise ValueError("Documents not indexed. Call index_documents first.")
        
        # Extract features for all documents
        features = []
        doc_ids = []
        
        for doc in self.documents:
            doc_features = self.feature_extractor.extract_features(query, doc)
            features.append(doc_features)
            doc_ids.append(doc['id'])
        
        features = np.array(features)
        
        # Predict scores
        scores = self.model.predict(features)
        
        # Get top k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = [
            (doc_ids[idx], float(scores[idx]))
            for idx in top_indices
        ]
        
        return results
    
    def _get_document_by_id(self, doc_id: str) -> Dict:
        """Get document by ID"""
        for doc in self.documents:
            if doc['id'] == doc_id:
                return doc
        return None
    
    def get_document_by_id(self, doc_id: str) -> Dict:
        """Public method to get document by ID"""
        return self._get_document_by_id(doc_id)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if not self.is_trained:
            return {}
        
        importance = self.model.feature_importances_
        feature_names = self.feature_extractor.get_feature_names()
        
        return dict(zip(feature_names, importance))
    
    def save_model(self, path: str = None) -> None:
        """Save the trained model"""
        if path is None:
            path = config.LAMBDAMART_MODEL_PATH
        
        model_data = {
            'model': self.model,
            'feature_extractor': self.feature_extractor,
            'documents': self.documents,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, path)
    
    def load_model(self, path: str = None) -> None:
        """Load a trained model"""
        if path is None:
            path = config.LAMBDAMART_MODEL_PATH
        
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.feature_extractor = model_data['feature_extractor']
        self.documents = model_data['documents']
        self.is_trained = model_data['is_trained']
