"""
Live Demo Platform for Search Ranking System
Interactive web interface to test and compare ranking algorithms
"""
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rankers.tfidf_ranker import TFIDFRanker
from src.rankers.bm25_ranker import BM25Ranker
from src.rankers.lambdamart_ranker import LambdaMARTRanker
import config

app = Flask(__name__)
CORS(app)

# Initialize rankers
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
        # Load documents
        with open(config.DOCUMENTS_PATH, 'r') as f:
            documents = json.load(f)
        
        # Initialize rankers
        rankers['tfidf'] = TFIDFRanker()
        rankers['bm25'] = BM25Ranker()
        rankers['lambdamart'] = LambdaMARTRanker()
        
        # Index documents
        rankers['tfidf'].index_documents(documents)
        rankers['bm25'].index_documents(documents)
        rankers['lambdamart'].index_documents(documents)
        
        # Try to load trained LambdaMART model
        try:
            rankers['lambdamart'].load_model()
        except:
            pass
        
        documents_loaded = True
        print("✓ Rankers initialized successfully")
        
    except Exception as e:
        print(f"Error initializing rankers: {e}")
        print("Please run: python scripts/generate_sample_data.py && python scripts/train_model.py")


@app.route('/')
def index():
    """Main demo page"""
    return render_template('index.html')


@app.route('/api/search', methods=['POST'])
def search():
    """Search endpoint for demo"""
    if not documents_loaded:
        return jsonify({'error': 'System not initialized. Run setup scripts first.'}), 500
    
    data = request.json
    query = data.get('query', '')
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
                            'score': round(score, 4),
                            'topic': doc['metadata'].get('topic', 'unknown'),
                            'quality_score': doc['metadata'].get('quality_score', 0)
                        })
                
                results[algo] = {
                    'algorithm': algo.upper(),
                    'results': formatted_results,
                    'count': len(formatted_results)
                }
            except Exception as e:
                results[algo] = {
                    'error': str(e)
                }
    
    return jsonify({
        'query': query,
        'results': results
    })


@app.route('/api/stats', methods=['GET'])
def stats():
    """Get system statistics"""
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


if __name__ == '__main__':
    print("="*60)
    print("Search Ranking System - Live Demo Platform")
    print("="*60)
    
    # Initialize rankers
    initialize_rankers()
    
    if documents_loaded:
        print("\n✓ System ready!")
        print(f"\n🌐 Open in browser: http://localhost:8080")
        print("\nPress Ctrl+C to stop the server")
        print("="*60 + "\n")
        
        app.run(host='0.0.0.0', port=8080, debug=False)
    else:
        print("\n❌ Failed to initialize system")
        print("\nPlease run:")
        print("  python scripts/generate_sample_data.py")
        print("  python scripts/train_model.py")
        print("\nThen start the demo again.")
        print("="*60)
