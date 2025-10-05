"""
Exploratory Analysis Notebook (Python Script Version)

This notebook explores the search ranking system's performance,
analyzes different algorithms, and visualizes results.

Run this file section by section or use with Jupyter by converting:
jupyter nbconvert --to notebook exploratory_analysis.py
"""

# %% [markdown]
# # Search Ranking System - Exploratory Analysis
# 
# This notebook analyzes:
# 1. Document and query distributions
# 2. Ranking algorithm performance
# 3. Feature importance analysis
# 4. A/B test results visualization

# %% Setup and Imports
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from src.rankers.tfidf_ranker import TFIDFRanker
from src.rankers.bm25_ranker import BM25Ranker
from src.rankers.lambdamart_ranker import LambdaMARTRanker
from src.evaluation.metrics import RankingMetrics

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("✓ Imports completed")

# %% [markdown]
# ## 1. Load and Explore Data

# %% Load Data
def load_data():
    """Load documents and queries"""
    with open('../data/sample_documents.json', 'r') as f:
        documents = json.load(f)
    
    with open('../data/sample_queries.json', 'r') as f:
        queries = json.load(f)
    
    return documents, queries

documents, queries = load_data()

print(f"Loaded {len(documents)} documents")
print(f"Loaded {len(queries)} queries")

# %% Document Statistics
doc_df = pd.DataFrame(documents)

print("\n=== Document Statistics ===")
print(f"Total documents: {len(doc_df)}")
print(f"\nDocument metadata fields:")
print(doc_df.columns.tolist())

# Expand metadata
metadata_df = pd.json_normalize(doc_df['metadata'])
doc_df = pd.concat([doc_df.drop('metadata', axis=1), metadata_df], axis=1)

print("\n=== Metadata Statistics ===")
print(doc_df[['quality_score', 'pagerank', 'freshness']].describe())

# %% Topic Distribution
topic_counts = doc_df['topic'].value_counts()

plt.figure(figsize=(10, 6))
topic_counts.plot(kind='bar', color='skyblue')
plt.title('Document Distribution by Topic', fontsize=16, fontweight='bold')
plt.xlabel('Topic', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('../data/topic_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✓ Saved: topic_distribution.png")

# %% Document Length Analysis
doc_df['content_length'] = doc_df['content'].str.split().str.len()
doc_df['title_length'] = doc_df['title'].str.split().str.len()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(doc_df['content_length'], bins=30, color='coral', edgecolor='black')
axes[0].set_title('Document Content Length Distribution', fontweight='bold')
axes[0].set_xlabel('Word Count')
axes[0].set_ylabel('Frequency')

axes[1].hist(doc_df['title_length'], bins=20, color='lightgreen', edgecolor='black')
axes[1].set_title('Document Title Length Distribution', fontweight='bold')
axes[1].set_xlabel('Word Count')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('../data/length_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✓ Saved: length_distribution.png")

# %% [markdown]
# ## 2. Query Analysis

# %% Query Statistics
query_df = pd.DataFrame(queries)

print("\n=== Query Statistics ===")
print(f"Total queries: {len(query_df)}")

query_df['query_length'] = query_df['query'].str.split().str.len()

print(f"\nQuery length statistics:")
print(query_df['query_length'].describe())

# Query topic distribution
query_topic_counts = query_df['topic'].value_counts()

plt.figure(figsize=(10, 6))
query_topic_counts.plot(kind='bar', color='lightcoral')
plt.title('Query Distribution by Topic', fontsize=16, fontweight='bold')
plt.xlabel('Topic', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('../data/query_topic_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✓ Saved: query_topic_distribution.png")

# %% [markdown]
# ## 3. Ranking Algorithm Comparison

# %% Initialize Rankers
print("\n=== Initializing Rankers ===")

tfidf_ranker = TFIDFRanker()
bm25_ranker = BM25Ranker()
lambdamart_ranker = LambdaMARTRanker()

# Index documents
tfidf_ranker.index_documents(documents)
bm25_ranker.index_documents(documents)
lambdamart_ranker.index_documents(documents)

print("✓ All rankers initialized and indexed")

# Try to load trained LambdaMART model
try:
    lambdamart_ranker.load_model('../models/lambdamart_model.pkl')
    print("✓ Loaded trained LambdaMART model")
except:
    print("⚠ LambdaMART model not trained, skipping LambdaMART analysis")

# %% Compare Rankings for Sample Queries
sample_queries = [
    "machine learning algorithms",
    "web development javascript",
    "database optimization",
    "cloud computing aws"
]

print("\n=== Sample Search Results ===\n")

for query in sample_queries:
    print(f"Query: '{query}'")
    print("-" * 60)
    
    tfidf_results = tfidf_ranker.search(query, top_k=3)
    bm25_results = bm25_ranker.search(query, top_k=3)
    
    print("TF-IDF Top 3:")
    for i, (doc_id, score) in enumerate(tfidf_results, 1):
        doc = tfidf_ranker.get_document_by_id(doc_id)
        print(f"  {i}. {doc['title'][:50]} (score: {score:.4f})")
    
    print("\nBM25 Top 3:")
    for i, (doc_id, score) in enumerate(bm25_results, 1):
        doc = bm25_ranker.get_document_by_id(doc_id)
        print(f"  {i}. {doc['title'][:50]} (score: {score:.4f})")
    
    print("\n")

# %% Score Distribution Analysis
test_query = "machine learning"

tfidf_results = tfidf_ranker.search(test_query, top_k=20)
bm25_results = bm25_ranker.search(test_query, top_k=20)

tfidf_scores = [score for _, score in tfidf_results]
bm25_scores = [score for _, score in bm25_results]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(range(1, len(tfidf_scores)+1), tfidf_scores, marker='o', color='blue', linewidth=2)
axes[0].set_title('TF-IDF Score Distribution', fontweight='bold')
axes[0].set_xlabel('Rank Position')
axes[0].set_ylabel('Relevance Score')
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, len(bm25_scores)+1), bm25_scores, marker='o', color='green', linewidth=2)
axes[1].set_title('BM25 Score Distribution', fontweight='bold')
axes[1].set_xlabel('Rank Position')
axes[1].set_ylabel('Relevance Score')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../data/score_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✓ Saved: score_distribution.png")

# %% [markdown]
# ## 4. Performance Metrics Evaluation

# %% Evaluate All Algorithms
print("\n=== Evaluating Ranking Quality ===\n")

# Create synthetic ground truth based on topic matching
evaluation_results = {}

for algo_name, ranker in [('TF-IDF', tfidf_ranker), ('BM25', bm25_ranker)]:
    all_rankings = []
    all_ground_truth = []
    
    for query_data in queries[:30]:
        query = query_data['query']
        query_topic = query_data['topic']
        
        results = ranker.search(query, top_k=10)
        all_rankings.append(results)
        
        # Ground truth: same topic = relevant
        ground_truth = {}
        for doc in documents:
            if doc['metadata']['topic'] == query_topic:
                ground_truth[doc['id']] = 3.0
            else:
                ground_truth[doc['id']] = 0.0
        
        all_ground_truth.append(ground_truth)
    
    # Calculate metrics
    metrics = RankingMetrics.evaluate_ranking(
        all_rankings,
        all_ground_truth,
        k_values=[1, 3, 5, 10]
    )
    
    evaluation_results[algo_name] = metrics
    
    print(f"{algo_name} Results:")
    for metric, value in sorted(metrics.items()):
        print(f"  {metric}: {value:.4f}")
    print()

# %% Visualize Metrics Comparison
metrics_to_plot = ['ndcg@10', 'map', 'mrr', 'precision@5']

comparison_data = []
for algo in evaluation_results:
    for metric in metrics_to_plot:
        if metric in evaluation_results[algo]:
            comparison_data.append({
                'Algorithm': algo,
                'Metric': metric,
                'Value': evaluation_results[algo][metric]
            })

comparison_df = pd.DataFrame(comparison_data)

plt.figure(figsize=(12, 6))
pivot_df = comparison_df.pivot(index='Metric', columns='Algorithm', values='Value')
pivot_df.plot(kind='bar', width=0.8)
plt.title('Algorithm Performance Comparison', fontsize=16, fontweight='bold')
plt.xlabel('Metric', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.legend(title='Algorithm', fontsize=10)
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1.0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('../data/algorithm_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✓ Saved: algorithm_comparison.png")

# %% [markdown]
# ## 5. Feature Importance (LambdaMART)

# %% Feature Importance Analysis
if lambdamart_ranker.is_trained:
    print("\n=== Feature Importance Analysis ===\n")
    
    feature_importance = lambdamart_ranker.get_feature_importance()
    
    # Sort by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Top 10 features
    top_features = sorted_features[:10]
    feature_names = [f[0] for f in top_features]
    importance_scores = [f[1] for f in top_features]
    
    plt.figure(figsize=(12, 6))
    plt.barh(feature_names, importance_scores, color='teal')
    plt.xlabel('Importance Score', fontsize=12)
    plt.title('Top 10 Most Important Features (LambdaMART)', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('../data/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Saved: feature_importance.png")
    
    print("\nTop 10 Features:")
    for i, (feature, score) in enumerate(top_features, 1):
        print(f"  {i:2d}. {feature:30s}: {score:.4f}")
else:
    print("\n⚠ LambdaMART not trained, skipping feature importance analysis")

# %% [markdown]
# ## 6. Summary Statistics

# %% Generate Summary Report
print("\n" + "="*60)
print("SUMMARY REPORT")
print("="*60)

print("\n📊 Dataset Statistics:")
print(f"  Total Documents: {len(documents)}")
print(f"  Total Queries: {len(queries)}")
print(f"  Topics: {len(doc_df['topic'].unique())}")
print(f"  Avg Doc Length: {doc_df['content_length'].mean():.1f} words")
print(f"  Avg Query Length: {query_df['query_length'].mean():.1f} words")

print("\n🏆 Best Algorithm Performance (nDCG@10):")
best_ndcg = max(evaluation_results.items(), key=lambda x: x[1].get('ndcg@10', 0))
print(f"  Winner: {best_ndcg[0]} with {best_ndcg[1]['ndcg@10']:.4f}")

print("\n📈 Key Metrics Achieved:")
for algo in evaluation_results:
    metrics = evaluation_results[algo]
    print(f"\n  {algo}:")
    print(f"    nDCG@10: {metrics.get('ndcg@10', 0):.4f}")
    print(f"    MAP: {metrics.get('map', 0):.4f}")
    print(f"    MRR: {metrics.get('mrr', 0):.4f}")
    print(f"    Precision@5: {metrics.get('precision@5', 0):.4f}")

print("\n✅ Analysis Complete!")
print("="*60)

# %% Save Results
results_summary = {
    'dataset_stats': {
        'num_documents': len(documents),
        'num_queries': len(queries),
        'num_topics': int(len(doc_df['topic'].unique())),
        'avg_doc_length': float(doc_df['content_length'].mean()),
        'avg_query_length': float(query_df['query_length'].mean())
    },
    'algorithm_performance': evaluation_results
}

with open('../data/analysis_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("\n✓ Saved analysis results to: data/analysis_results.json")

# %% [markdown]
# ## End of Analysis
# 
# Generated visualizations:
# - topic_distribution.png
# - length_distribution.png
# - query_topic_distribution.png
# - score_distribution.png
# - algorithm_comparison.png
# - feature_importance.png
# 
# Generated data:
# - analysis_results.json
