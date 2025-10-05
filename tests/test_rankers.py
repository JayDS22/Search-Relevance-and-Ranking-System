"""
Unit tests for ranking algorithms
"""
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rankers.tfidf_ranker import TFIDFRanker
from src.rankers.bm25_ranker import BM25Ranker
from src.rankers.lambdamart_ranker import LambdaMARTRanker


# Sample test documents
TEST_DOCUMENTS = [
    {
        'id': 'doc1',
        'title': 'Machine Learning Basics',
        'content': 'Machine learning is a subset of artificial intelligence that focuses on algorithms.',
        'metadata': {'quality_score': 0.8, 'pagerank': 0.7, 'freshness': 0.9}
    },
    {
        'id': 'doc2',
        'title': 'Deep Learning Tutorial',
        'content': 'Deep learning uses neural networks with multiple layers to learn complex patterns.',
        'metadata': {'quality_score': 0.9, 'pagerank': 0.8, 'freshness': 0.8}
    },
    {
        'id': 'doc3',
        'title': 'Web Development Guide',
        'content': 'Learn web development with HTML, CSS, and JavaScript to build modern websites.',
        'metadata': {'quality_score': 0.7, 'pagerank': 0.6, 'freshness': 0.7}
    },
    {
        'id': 'doc4',
        'title': 'Database Management',
        'content': 'Database management systems help organize and retrieve data efficiently.',
        'metadata': {'quality_score': 0.75, 'pagerank': 0.65, 'freshness': 0.6}
    }
]


class TestTFIDFRanker:
    """Test TF-IDF ranker"""
    
    def test_initialization(self):
        """Test ranker initialization"""
        ranker = TFIDFRanker()
        assert ranker is not None
        assert ranker.doc_vectors is None
    
    def test_indexing(self):
        """Test document indexing"""
        ranker = TFIDFRanker()
        ranker.index_documents(TEST_DOCUMENTS)
        
        assert len(ranker.documents) == len(TEST_DOCUMENTS)
        assert ranker.doc_vectors is not None
    
    def test_search(self):
        """Test search functionality"""
        ranker = TFIDFRanker()
        ranker.index_documents(TEST_DOCUMENTS)
        
        results = ranker.search('machine learning', top_k=2)
        
        assert len(results) <= 2
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 2 for r in results)
        
        # Check that scores are floats
        for doc_id, score in results:
            assert isinstance(doc_id, str)
            assert isinstance(score, float)
    
    def test_relevant_results(self):
        """Test that relevant documents rank higher"""
        ranker = TFIDFRanker()
        ranker.index_documents(TEST_DOCUMENTS)
        
        results = ranker.search('machine learning', top_k=4)
        
        # First result should be related to machine learning
        top_doc_id = results[0][0]
        top_doc = ranker.get_document_by_id(top_doc_id)
        
        assert 'machine' in top_doc['content'].lower() or 'learning' in top_doc['content'].lower()
    
    def test_get_document_by_id(self):
        """Test document retrieval"""
        ranker = TFIDFRanker()
        ranker.index_documents(TEST_DOCUMENTS)
        
        doc = ranker.get_document_by_id('doc1')
        assert doc is not None
        assert doc['id'] == 'doc1'
        
        doc = ranker.get_document_by_id('nonexistent')
        assert doc is None


class TestBM25Ranker:
    """Test BM25 ranker"""
    
    def test_initialization(self):
        """Test ranker initialization"""
        ranker = BM25Ranker()
        assert ranker is not None
        assert ranker.k1 == 1.5
        assert ranker.b == 0.75
    
    def test_custom_parameters(self):
        """Test custom BM25 parameters"""
        ranker = BM25Ranker(k1=2.0, b=0.5)
        assert ranker.k1 == 2.0
        assert ranker.b == 0.5
    
    def test_indexing(self):
        """Test document indexing"""
        ranker = BM25Ranker()
        ranker.index_documents(TEST_DOCUMENTS)
        
        assert len(ranker.documents) == len(TEST_DOCUMENTS)
        assert ranker.N == len(TEST_DOCUMENTS)
        assert ranker.avgdl > 0
    
    def test_search(self):
        """Test search functionality"""
        ranker = BM25Ranker()
        ranker.index_documents(TEST_DOCUMENTS)
        
        results = ranker.search('web development', top_k=3)
        
        assert len(results) <= 3
        assert all(isinstance(r, tuple) for r in results)
        
        # Check that scores are positive
        for doc_id, score in results:
            assert score > 0
    
    def test_idf_calculation(self):
        """Test IDF scores are calculated"""
        ranker = BM25Ranker()
        ranker.index_documents(TEST_DOCUMENTS)
        
        assert len(ranker.idf) > 0
        assert all(isinstance(v, float) for v in ranker.idf.values())


class TestLambdaMARTRanker:
    """Test LambdaMART ranker"""
    
    def test_initialization(self):
        """Test ranker initialization"""
        ranker = LambdaMARTRanker()
        assert ranker is not None
        assert not ranker.is_trained
    
    def test_indexing(self):
        """Test document indexing"""
        ranker = LambdaMARTRanker()
        ranker.index_documents(TEST_DOCUMENTS)
        
        assert len(ranker.documents) == len(TEST_DOCUMENTS)
    
    def test_training(self):
        """Test model training"""
        ranker = LambdaMARTRanker()
        ranker.index_documents(TEST_DOCUMENTS)
        
        # Create simple training data
        training_data = [
            {
                'query': 'machine learning',
                'documents': ['doc1', 'doc2', 'doc3'],
                'relevance_scores': [3, 2, 0]
            },
            {
                'query': 'web development',
                'documents': ['doc3', 'doc1', 'doc4'],
                'relevance_scores': [3, 0, 1]
            }
        ]
        
        ranker.train(training_data)
        
        assert ranker.is_trained
    
    def test_search_requires_training(self):
        """Test that search requires trained model"""
        ranker = LambdaMARTRanker()
        ranker.index_documents(TEST_DOCUMENTS)
        
        with pytest.raises(ValueError):
            ranker.search('test query')
    
    def test_feature_extraction(self):
        """Test feature extraction"""
        ranker = LambdaMARTRanker()
        ranker.index_documents(TEST_DOCUMENTS)
        
        features = ranker.feature_extractor.extract_features('machine learning', TEST_DOCUMENTS[0])
        
        assert isinstance(features, type(pytest.importorskip('numpy').array([])))
        assert len(features) > 0


def test_ranker_comparison():
    """Test that different rankers can work with same documents"""
    tfidf = TFIDFRanker()
    bm25 = BM25Ranker()
    
    tfidf.index_documents(TEST_DOCUMENTS)
    bm25.index_documents(TEST_DOCUMENTS)
    
    query = 'machine learning algorithms'
    
    tfidf_results = tfidf.search(query, top_k=3)
    bm25_results = bm25.search(query, top_k=3)
    
    # Both should return results
    assert len(tfidf_results) > 0
    assert len(bm25_results) > 0
    
    # Results might be different but should have same format
    assert isinstance(tfidf_results[0], tuple)
    assert isinstance(bm25_results[0], tuple)
