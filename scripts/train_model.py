"""
Train ranking models
"""
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rankers.tfidf_ranker import TFIDFRanker
from src.rankers.bm25_ranker import BM25Ranker
from src.rankers.lambdamart_ranker import LambdaMARTRanker
import config


def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def train_tfidf():
    """Train and save TF-IDF model"""
    print("\n=== Training TF-IDF Ranker ===")
    
    # Load documents
    documents = load_json(config.DOCUMENTS_PATH)
    print(f"Loaded {len(documents)} documents")
    
    # Initialize and train
    ranker = TFIDFRanker()
    ranker.index_documents(documents)
    
    # Save model
    ranker.save_model()
    print(f"Model saved to {config.TFIDF_MODEL_PATH}")
    
    # Test search
    test_query = "machine learning algorithms"
    results = ranker.search(test_query, top_k=5)
    print(f"\nTest search for '{test_query}':")
    for doc_id, score in results[:3]:
        doc = ranker.get_document_by_id(doc_id)
        print(f"  {doc_id}: {doc['title']} (score: {score:.4f})")


def train_bm25():
    """Train and save BM25 model"""
    print("\n=== Training BM25 Ranker ===")
    
    # Load documents
    documents = load_json(config.DOCUMENTS_PATH)
    print(f"Loaded {len(documents)} documents")
    
    # Initialize and train
    ranker = BM25Ranker()
    ranker.index_documents(documents)
    
    # Save model
    ranker.save_model()
    print(f"Model saved to {config.BM25_MODEL_PATH}")
    
    # Test search
    test_query = "web development javascript"
    results = ranker.search(test_query, top_k=5)
    print(f"\nTest search for '{test_query}':")
    for doc_id, score in results[:3]:
        doc = ranker.get_document_by_id(doc_id)
        print(f"  {doc_id}: {doc['title']} (score: {score:.4f})")


def train_lambdamart():
    """Train and save LambdaMART model"""
    print("\n=== Training LambdaMART Ranker ===")
    
    # Load data
    documents = load_json(config.DOCUMENTS_PATH)
    training_data = load_json(config.TRAINING_DATA_PATH)
    
    print(f"Loaded {len(documents)} documents")
    print(f"Loaded {len(training_data)} training samples")
    
    # Initialize ranker
    ranker = LambdaMARTRanker()
    ranker.index_documents(documents)
    
    # Train model
    print("Training model (this may take a few minutes)...")
    ranker.train(training_data)
    
    # Save model
    ranker.save_model()
    print(f"Model saved to {config.LAMBDAMART_MODEL_PATH}")
    
    # Show feature importance
    feature_importance = ranker.get_feature_importance()
    print("\nTop 10 Most Important Features:")
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_features[:10]:
        print(f"  {feature}: {importance:.4f}")
    
    # Test search
    test_query = "database optimization"
    results = ranker.search(test_query, top_k=5)
    print(f"\nTest search for '{test_query}':")
    for doc_id, score in results[:3]:
        doc = ranker.get_document_by_id(doc_id)
        print(f"  {doc_id}: {doc['title']} (score: {score:.4f})")


def main():
    """Train all models"""
    print("Starting model training...")
    
    # Create models directory
    config.MODELS_DIR.mkdir(exist_ok=True)
    
    # Check if data exists
    if not config.DOCUMENTS_PATH.exists():
        print(f"Error: Documents file not found at {config.DOCUMENTS_PATH}")
        print("Please run generate_sample_data.py first")
        return
    
    # Train models
    try:
        train_tfidf()
        train_bm25()
        
        # LambdaMART requires training data
        if config.TRAINING_DATA_PATH.exists():
            train_lambdamart()
        else:
            print("\nSkipping LambdaMART training (no training data found)")
            print("Run generate_sample_data.py to create training data")
        
        print("\n" + "="*50)
        print("Training complete! All models saved.")
        print("="*50)
        
    except Exception as e:
        print(f"\nError during training: {e}")
        raise


if __name__ == '__main__':
    main()
