"""
Live Demo Platform for Search Ranking System
Interactive web interface to test and compare ranking algorithms
"""
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import os
import json
import traceback

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rankers.tfidf_ranker import TFIDFRanker
from src.rankers.bm25_ranker import BM25Ranker
from src.rankers.lambdamart_ranker import LambdaMARTRanker
import config

app = Flask(__name__)
CORS(app)

# Global variables
rankers = {
    'tfidf': None,
    'bm25': None,
    'lambdamart': None
}
documents_loaded = False

def initialize_rankers():
    """Initialize and load rankers"""
    global rankers, documents_loaded
    
    try:
        print("Initializing rankers...")
        
        # Check if data file exists
        if not config.DOCUMENTS_PATH.exists():
            print(f"❌ Error: {config.DOCUMENTS_PATH} not found!")
            print("Please run: python scripts/generate_sample_data.py")
            return False
        
        # Load documents
        print(f"Loading documents from {config.DOCUMENTS_PATH}...")
        with open(config.DOCUMENTS_PATH, 'r') as f:
            documents = json.load(f)
        
        print(f"Loaded {len(documents)} documents")
        
        # Initialize rankers
        print("Creating ranker instances...")
        rankers['tfidf'] = TFIDFRanker()
        rankers['bm25'] = BM25Ranker()
        rankers['lambdamart'] = LambdaMARTRanker()
        
        # Index documents
        print("Indexing documents...")
        rankers['tfidf'].index_documents(documents)
        rankers['bm25'].index_documents(documents)
        rankers['lambdamart'].index_documents(documents)
        
        # Try to load trained LambdaMART model
        try:
            if config.LAMBDAMART_MODEL_PATH.exists():
                print("Loading trained LambdaMART model...")
                rankers['lambdamart'].load_model()
                print("✓ LambdaMART model loaded")
            else:
                print("⚠ LambdaMART model not found (searches will fail for this algorithm)")
        except Exception as e:
            print(f"⚠ Could not load LambdaMART model: {e}")
        
        documents_loaded = True
        print("✓ All rankers initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing rankers: {e}")
        traceback.print_exc()
        return False


@app.route('/')
def index():
    """Main demo page"""
    return render_template('index.html')


@app.route('/api/search', methods=['POST'])
def search():
    """Search endpoint for demo"""
    try:
        if not documents_loaded:
            return jsonify({
                'error': 'System not initialized. Please run: python scripts/generate_sample_data.py && python scripts/train_model.py'
            }), 500
        
        data = request.json
        query = data.get('query', '').strip()
        algorithms = data.get('algorithms', ['tfidf', 'bm25', 'lambdamart'])
        top_k = min(data.get('top_k', 10), 20)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        results = {}
        
        for algo in algorithms:
            if algo in rankers and rankers[algo]:
                try:
                    search_results = rankers[algo].search(query, top_k=top_k)
                    
                    # Format results
                    formatted_results = []
                    for doc_id, score in search_results:
                        doc = rankers[algo].get_document_by_id(doc_id)
                        if doc:
                            formatted_results.append({
                                'id': doc_id,
                                'title': doc['title'],
                                'content': doc['content'][:200] + '...' if len(doc['content']) > 200 else doc['content'],
                                'score': round(float(score), 4),
                                'topic': doc['metadata'].get('topic', 'unknown'),
                                'quality_score': float(doc['metadata'].get('quality_score', 0))
                            })
                    
                    results[algo] = {
                        'algorithm': algo.upper(),
                        'results': formatted_results,
                        'count': len(formatted_results)
                    }
                except Exception as e:
                    print(f"Error searching with {algo}: {e}")
                    traceback.print_exc()
                    results[algo] = {
                        'error': str(e),
                        'results': [],
                        'count': 0
                    }
        
        return jsonify({
            'query': query,
            'results': results
        })
    
    except Exception as e:
        print(f"Error in search endpoint: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def stats():
    """Get system statistics"""
    try:
        if not documents_loaded:
            return jsonify({'error': 'System not initialized'}), 500
        
        stats_data = {
            'indexed_documents': len(rankers['bm25'].documents) if rankers['bm25'] else 0,
            'available_algorithms': 3,
            'algorithms': {
                'tfidf': {'name': 'TF-IDF', 'description': 'Classic baseline algorithm', 'speed': 'Fast'},
                'bm25': {'name': 'BM25', 'description': 'Probabilistic retrieval model', 'speed': 'Fast'},
                'lambdamart': {'name': 'LambdaMART', 'description': 'Machine learning ranking', 'speed': 'Medium'}
            }
        }
        
        return jsonify(stats_data)
    
    except Exception as e:
        print(f"Error in stats endpoint: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/sample-queries', methods=['GET'])
def sample_queries():
    """Get sample queries for demo"""
    samples = [
        "machine learning algorithms",
        "web development javascript",
        "database optimization",
        "cloud computing aws",
        "python programming",
        "data science analytics",
        "neural networks deep learning",
        "REST API design",
        "docker kubernetes",
        "sql query performance"
    ]
    
    return jsonify({'queries': samples})


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if documents_loaded else 'not_initialized',
        'documents_loaded': documents_loaded
    })


if __name__ == '__main__':
    print("="*60)
    print("Search Ranking System - Live Demo Platform")
    print("="*60)
    print()
    
    # Initialize rankers
    success = initialize_rankers()
    
    if success:
        print()
        print("="*60)
        print("✓ System ready!")
        print("="*60)
        print()
        print("🌐 Open in browser: http://localhost:8080")
        print()
        print("Press Ctrl+C to stop the server")
        print("="*60)
        print()
        
        app.run(host='0.0.0.0', port=8080, debug=True)
    else:
        print()
        print("="*60)
        print("❌ Failed to initialize system")
        print("="*60)
        print()
        print("Please run these commands first:")
        print("  1. python scripts/generate_sample_data.py")
        print("  2. python scripts/train_model.py")
        print()
        print("Then start the demo again:")
        print("  python demo/app.py")
        print("="*60)
