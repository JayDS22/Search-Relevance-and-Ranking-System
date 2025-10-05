"""
Run A/B testing experiments
"""
import json
import sys
import os
import random

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rankers.tfidf_ranker import TFIDFRanker
from src.rankers.bm25_ranker import BM25Ranker
from src.rankers.lambdamart_ranker import LambdaMARTRanker
from src.evaluation.metrics import RankingMetrics
from src.ab_testing.experiment import ABTestExperiment
import config


def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def simulate_user_engagement(results, doc_id):
    """
    Simulate user engagement metric (e.g., click-through)
    Higher ranked relevant docs get higher engagement
    """
    # Find position of document
    position = None
    for i, (rid, _) in enumerate(results):
        if rid == doc_id:
            position = i
            break
    
    if position is None:
        return 0.0
    
    # Position-based engagement (higher position = higher engagement)
    # Simulating click-through rate that decreases with position
    base_engagement = 1.0 / (position + 1)
    
    # Add some randomness
    noise = random.uniform(-0.1, 0.1)
    
    return max(0, min(1, base_engagement + noise))


def run_experiment(experiment_name, baseline_ranker, treatment_ranker, queries, documents):
    """Run a single A/B test experiment"""
    print(f"\n{'='*60}")
    print(f"Running Experiment: {experiment_name}")
    print(f"{'='*60}")
    
    # Create experiment
    experiment = ABTestExperiment(
        experiment_id=experiment_name.lower().replace(' ', '_'),
        baseline_name=baseline_ranker.__class__.__name__,
        treatment_name=treatment_ranker.__class__.__name__,
        traffic_split=0.5
    )
    
    # Simulate user queries
    num_users = 2000
    print(f"Simulating {num_users} users...")
    
    for user_id in range(num_users):
        # Assign to variant
        variant = experiment.assign_variant(f"user_{user_id}")
        
        # Choose random query
        query_data = random.choice(queries)
        query = query_data['query']
        
        # Get search results based on variant
        if variant == 'baseline':
            results = baseline_ranker.search(query, top_k=10)
        else:
            results = treatment_ranker.search(query, top_k=10)
        
        # Simulate user interaction
        if results:
            # User clicks on first relevant document
            clicked_doc_id = results[0][0]
            engagement = simulate_user_engagement(results, clicked_doc_id)
            
            # Record metric
            experiment.record_metric(variant, engagement)
    
    # Calculate results
    print("\nCalculating statistics...")
    results = experiment.calculate_statistics()
    
    # Print results
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS")
    print("="*60)
    
    print(f"\nBaseline ({results['baseline']['name']}):")
    print(f"  Sample size: {results['baseline']['n']}")
    print(f"  Mean engagement: {results['baseline']['mean']:.4f}")
    print(f"  Std deviation: {results['baseline']['std']:.4f}")
    
    print(f"\nTreatment ({results['treatment']['name']}):")
    print(f"  Sample size: {results['treatment']['n']}")
    print(f"  Mean engagement: {results['treatment']['mean']:.4f}")
    print(f"  Std deviation: {results['treatment']['std']:.4f}")
    
    print(f"\nStatistical Analysis:")
    print(f"  t-statistic: {results['statistics']['t_statistic']:.4f}")
    print(f"  p-value: {results['statistics']['p_value']:.6f}")
    print(f"  Cohen's d: {results['statistics']['cohens_d']:.4f}")
    print(f"  95% CI: [{results['statistics']['ci_95_lower']:.4f}, {results['statistics']['ci_95_upper']:.4f}]")
    
    print(f"\nResults:")
    print(f"  Absolute improvement: {results['results']['absolute_improvement']:.4f}")
    print(f"  Relative improvement: {results['results']['relative_improvement_pct']:.2f}%")
    print(f"  Statistical significance: {'YES' if results['results']['is_significant'] else 'NO'}")
    print(f"  Winner: {results['results']['winner']}")
    
    # Save results
    output_file = config.DATA_DIR / f"{experiment_name.lower().replace(' ', '_')}_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    return results


def evaluate_ranking_quality(ranker, ranker_name, queries, documents):
    """Evaluate ranking quality using standard metrics"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {ranker_name}")
    print(f"{'='*60}")
    
    # Create ground truth based on topic matching
    all_rankings = []
    all_ground_truth = []
    
    for query_data in queries[:30]:  # Use subset for evaluation
        query = query_data['query']
        query_topic = query_data['topic']
        
        # Get ranking
        results = ranker.search(query, top_k=20)
        all_rankings.append(results)
        
        # Create ground truth (docs from same topic are relevant)
        ground_truth = {}
        for doc in documents:
            if doc['metadata']['topic'] == query_topic:
                ground_truth[doc['id']] = 3.0  # Relevant
            else:
                ground_truth[doc['id']] = 0.0  # Not relevant
        
        all_ground_truth.append(ground_truth)
    
    # Calculate metrics
    metrics = RankingMetrics.evaluate_ranking(
        all_rankings,
        all_ground_truth,
        k_values=[1, 3, 5, 10]
    )
    
    # Print metrics
    print("\nRanking Quality Metrics:")
    for metric_name, value in sorted(metrics.items()):
        print(f"  {metric_name}: {value:.4f}")
    
    return metrics


def main():
    """Main experiment runner"""
    print("Search Ranking System - Experiment Runner")
    print("="*60)
    
    # Load data
    print("Loading data...")
    documents = load_json(config.DOCUMENTS_PATH)
    queries = load_json(config.QUERIES_PATH)
    
    print(f"Loaded {len(documents)} documents")
    print(f"Loaded {len(queries)} queries")
    
    # Initialize rankers
    print("\nInitializing rankers...")
    
    tfidf_ranker = TFIDFRanker()
    bm25_ranker = BM25Ranker()
    lambdamart_ranker = LambdaMARTRanker()
    
    # Index documents
    tfidf_ranker.index_documents(documents)
    bm25_ranker.index_documents(documents)
    lambdamart_ranker.index_documents(documents)
    
    # Try to load trained LambdaMART model
    if config.LAMBDAMART_MODEL_PATH.exists():
        print("Loading pre-trained LambdaMART model...")
        lambdamart_ranker.load_model()
    else:
        print("Warning: LambdaMART model not found. Using untrained model.")
        print("Run train_model.py first for better results.")
    
    # Evaluate ranking quality
    print("\n" + "="*60)
    print("PART 1: RANKING QUALITY EVALUATION")
    print("="*60)
    
    tfidf_metrics = evaluate_ranking_quality(tfidf_ranker, "TF-IDF", queries, documents)
    bm25_metrics = evaluate_ranking_quality(bm25_ranker, "BM25", queries, documents)
    
    if lambdamart_ranker.is_trained:
        lambdamart_metrics = evaluate_ranking_quality(lambdamart_ranker, "LambdaMART", queries, documents)
    
    # Run A/B tests
    print("\n" + "="*60)
    print("PART 2: A/B TESTING EXPERIMENTS")
    print("="*60)
    
    # Experiment 1: TF-IDF vs BM25
    exp1_results = run_experiment(
        "TF-IDF vs BM25",
        tfidf_ranker,
        bm25_ranker,
        queries,
        documents
    )
    
    # Experiment 2: BM25 vs LambdaMART (if trained)
    if lambdamart_ranker.is_trained:
        exp2_results = run_experiment(
            "BM25 vs LambdaMART",
            bm25_ranker,
            lambdamart_ranker,
            queries,
            documents
        )
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    print("\nRanking Quality (nDCG@10):")
    print(f"  TF-IDF: {tfidf_metrics.get('ndcg@10', 0):.4f}")
    print(f"  BM25: {bm25_metrics.get('ndcg@10', 0):.4f}")
    if lambdamart_ranker.is_trained:
        print(f"  LambdaMART: {lambdamart_metrics.get('ndcg@10', 0):.4f}")
    
    print("\nA/B Test Results:")
    print(f"  TF-IDF vs BM25: {exp1_results['results']['relative_improvement_pct']:.2f}% improvement")
    if lambdamart_ranker.is_trained:
        print(f"  BM25 vs LambdaMART: {exp2_results['results']['relative_improvement_pct']:.2f}% improvement")
    
    print("\n" + "="*60)
    print("All experiments complete!")
    print("="*60)


if __name__ == '__main__':
    main()
