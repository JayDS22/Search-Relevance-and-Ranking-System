"""
Flask API for Search Ranking System
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.rankers.tfidf_ranker import TFIDFRanker
from src.rankers.bm25_ranker import BM25Ranker
from src.rankers.lambdamart_ranker import LambdaMARTRanker
from src.evaluation.metrics import RankingMetrics
from src.ab_testing.experiment import ABTestManager
import config

app = Flask(__name__)
CORS(app)

# Initialize rankers
rankers = {
    'tfidf': TFIDFRanker(),
    'bm25': BM25Ranker(),
    'lambdamart': LambdaMARTRanker()
}

# Initialize A/B test manager
ab_test_manager = ABTestManager()

# Store documents in memory (in production, use a database)
indexed_documents = []


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'search-ranking-system',
        'version': '1.0.0'
    })


@app.route('/index', methods=['POST'])
def index_documents():
    """
    Index documents for search
    
    Expected JSON:
    {
        "documents": [
            {
                "id": "doc1",
                "title": "Document Title",
                "content": "Document content...",
                "metadata": {...}
            }
        ]
    }
    """
    try:
        data = request.get_json()
        
        if 'documents' not in data:
            return jsonify({'error': 'Missing documents field'}), 400
        
        documents = data['documents']
        
        # Validate documents
        for doc in documents:
            if 'id' not in doc or 'content' not in doc:
                return jsonify({'error': 'Each document must have id and content'}), 400
        
        # Store documents
        global indexed_documents
        indexed_documents = documents
        
        # Index in all rankers
        rankers['tfidf'].index_documents(documents)
        rankers['bm25'].index_documents(documents)
        rankers['lambdamart'].index_documents(documents)
        
        return jsonify({
            'status': 'success',
            'indexed_count': len(documents),
            'message': f'Indexed {len(documents)} documents'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/search', methods=['POST'])
def search():
    """
    Search documents
    
    Expected JSON:
    {
        "query": "search query",
        "algorithm": "tfidf|bm25|lambdamart",
        "top_k": 10,
        "user_id": "optional_user_id",
        "experiment_id": "optional_experiment_id"
    }
    """
    try:
        data = request.get_json()
        
        # Validate request
        if 'query' not in data:
            return jsonify({'error': 'Missing query field'}), 400
        
        query = data['query']
        algorithm = data.get('algorithm', 'bm25')
        top_k = min(data.get('top_k', config.DEFAULT_TOP_K), config.MAX_TOP_K)
        user_id = data.get('user_id')
        experiment_id = data.get('experiment_id')
        
        # Check if documents are indexed
        if not indexed_documents:
            return jsonify({'error': 'No documents indexed. Call /index first'}), 400
        
        # Handle A/B testing
        if experiment_id and user_id:
            experiment = ab_test_manager.get_experiment(experiment_id)
            if experiment:
                variant = experiment.assign_variant(user_id)
                if variant == 'baseline':
                    algorithm = experiment.baseline_name
                else:
                    algorithm = experiment.treatment_name
        
        # Validate algorithm
        if algorithm not in rankers:
            return jsonify({
                'error': f'Invalid algorithm. Choose from: {list(rankers.keys())}'
            }), 400
        
        # Perform search
        ranker = rankers[algorithm]
        results = ranker.search(query, top_k)
        
        # Get full document details
        ranked_docs = []
        for doc_id, score in results:
            doc = ranker.get_document_by_id(doc_id)
            if doc:
                ranked_docs.append({
                    'id': doc_id,
                    'title': doc.get('title', ''),
                    'content': doc.get('content', '')[:200] + '...',  # Preview
                    'score': score,
                    'metadata': doc.get('metadata', {})
                })
        
        return jsonify({
            'query': query,
            'algorithm': algorithm,
            'total_results': len(ranked_docs),
            'results': ranked_docs
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/evaluate', methods=['POST'])
def evaluate():
    """
    Evaluate ranking quality
    
    Expected JSON:
    {
        "queries": [
            {
                "query": "search query",
                "algorithm": "bm25",
                "ground_truth": {
                    "doc1": 3.0,
                    "doc2": 2.0,
                    "doc3": 1.0
                }
            }
        ]
    }
    """
    try:
        data = request.get_json()
        
        if 'queries' not in data:
            return jsonify({'error': 'Missing queries field'}), 400
        
        queries = data['queries']
        
        # Collect results for evaluation
        predicted_rankings = []
        ground_truths = []
        
        for query_item in queries:
            query = query_item['query']
            algorithm = query_item.get('algorithm', 'bm25')
            ground_truth = query_item['ground_truth']
            
            # Get ranking
            ranker = rankers[algorithm]
            results = ranker.search(query, top_k=20)
            
            predicted_rankings.append(results)
            ground_truths.append(ground_truth)
        
        # Calculate metrics
        metrics = RankingMetrics.evaluate_ranking(
            predicted_rankings,
            ground_truths
        )
        
        return jsonify({
            'metrics': metrics,
            'num_queries': len(queries)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/experiment/create', methods=['POST'])
def create_experiment():
    """
    Create an A/B test experiment
    
    Expected JSON:
    {
        "experiment_id": "exp_001",
        "baseline": "tfidf",
        "treatment": "bm25",
        "traffic_split": 0.5
    }
    """
    try:
        data = request.get_json()
        
        experiment_id = data.get('experiment_id')
        baseline = data.get('baseline')
        treatment = data.get('treatment')
        traffic_split = data.get('traffic_split', 0.5)
        
        if not experiment_id or not baseline or not treatment:
            return jsonify({'error': 'Missing required fields'}), 400
        
        experiment = ab_test_manager.create_experiment(
            experiment_id,
            baseline,
            treatment,
            traffic_split
        )
        
        return jsonify({
            'status': 'success',
            'experiment_id': experiment_id,
            'baseline': baseline,
            'treatment': treatment,
            'traffic_split': traffic_split
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/experiment/<experiment_id>/record', methods=['POST'])
def record_experiment_metric(experiment_id):
    """
    Record a metric for an experiment
    
    Expected JSON:
    {
        "user_id": "user123",
        "metric_value": 0.85
    }
    """
    try:
        experiment = ab_test_manager.get_experiment(experiment_id)
        
        if not experiment:
            return jsonify({'error': 'Experiment not found'}), 404
        
        data = request.get_json()
        user_id = data.get('user_id')
        metric_value = data.get('metric_value')
        
        if user_id is None or metric_value is None:
            return jsonify({'error': 'Missing user_id or metric_value'}), 400
        
        # Get variant assignment
        variant = experiment.assign_variant(user_id)
        
        # Record metric
        experiment.record_metric(variant, metric_value)
        
        return jsonify({
            'status': 'success',
            'variant': variant
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/experiment/<experiment_id>/results', methods=['GET'])
def get_experiment_results(experiment_id):
    """Get A/B test results"""
    try:
        experiment = ab_test_manager.get_experiment(experiment_id)
        
        if not experiment:
            return jsonify({'error': 'Experiment not found'}), 404
        
        # Check if ready for analysis
        ready, message = experiment.is_ready_for_analysis()
        
        if not ready:
            return jsonify({
                'status': 'insufficient_data',
                'message': message
            })
        
        # Calculate statistics
        results = experiment.calculate_statistics()
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/experiments', methods=['GET'])
def list_experiments():
    """List all experiments"""
    try:
        experiment_ids = ab_test_manager.list_experiments()
        
        return jsonify({
            'experiments': experiment_ids,
            'count': len(experiment_ids)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/documents', methods=['GET'])
def get_documents():
    """Get all indexed documents"""
    return jsonify({
        'documents': indexed_documents,
        'count': len(indexed_documents)
    })


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    return jsonify({
        'indexed_documents': len(indexed_documents),
        'available_algorithms': list(rankers.keys()),
        'active_experiments': len(ab_test_manager.list_experiments())
    })


if __name__ == '__main__':
    print("Starting Search Ranking System API...")
    print(f"Available at http://{config.API_HOST}:{config.API_PORT}")
    app.run(
        host=config.API_HOST,
        port=config.API_PORT,
        debug=config.DEBUG_MODE
    )
