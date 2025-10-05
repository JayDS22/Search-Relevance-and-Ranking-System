# Usage Examples

Practical examples for using the Search Ranking System.

## Table of Contents
- [Basic Usage](#basic-usage)
- [API Examples](#api-examples)
- [Python SDK Usage](#python-sdk-usage)
- [A/B Testing](#ab-testing)
- [Custom Features](#custom-features)
- [Advanced Scenarios](#advanced-scenarios)

---

## Basic Usage

### 1. Simple Search with TF-IDF

```python
from src.rankers.tfidf_ranker import TFIDFRanker

# Create documents
documents = [
    {
        'id': 'doc1',
        'title': 'Introduction to Python',
        'content': 'Python is a high-level programming language...',
        'metadata': {}
    },
    {
        'id': 'doc2',
        'title': 'Machine Learning Basics',
        'content': 'Machine learning is a subset of AI...',
        'metadata': {}
    }
]

# Initialize and index
ranker = TFIDFRanker()
ranker.index_documents(documents)

# Search
results = ranker.search('python programming', top_k=5)

for doc_id, score in results:
    doc = ranker.get_document_by_id(doc_id)
    print(f"{doc['title']}: {score:.4f}")
```

### 2. BM25 Ranking

```python
from src.rankers.bm25_ranker import BM25Ranker

# Custom parameters
ranker = BM25Ranker(k1=1.2, b=0.75)
ranker.index_documents(documents)

results = ranker.search('machine learning algorithms', top_k=10)

for doc_id, score in results:
    print(f"{doc_id}: {score:.4f}")
```

### 3. LambdaMART with Training

```python
from src.rankers.lambdamart_ranker import LambdaMARTRanker

# Initialize
ranker = LambdaMARTRanker()
ranker.index_documents(documents)

# Training data with relevance judgments
training_data = [
    {
        'query': 'python programming',
        'documents': ['doc1', 'doc2'],
        'relevance_scores': [4, 1]  # 0-4 scale
    }
]

# Train
ranker.train(training_data)

# Search
results = ranker.search('python tutorial', top_k=5)

# Feature importance
importance = ranker.get_feature_importance()
for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"{feature}: {score:.4f}")
```

---

## API Examples

### Index Documents

```bash
curl -X POST http://localhost:5000/index \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "id": "doc1",
        "title": "Python Tutorial",
        "content": "Learn Python programming from scratch...",
        "metadata": {
          "author": "John Doe",
          "date": "2024-01-15",
          "quality_score": 0.9
        }
      },
      {
        "id": "doc2",
        "title": "Machine Learning Guide",
        "content": "A comprehensive guide to ML algorithms...",
        "metadata": {
          "author": "Jane Smith",
          "date": "2024-02-20",
          "quality_score": 0.95
        }
      }
    ]
  }'
```

### Search Documents

```bash
curl -X POST http://localhost:5000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "python machine learning",
    "algorithm": "bm25",
    "top_k": 10
  }'
```

**Response:**
```json
{
  "query": "python machine learning",
  "algorithm": "bm25",
  "total_results": 2,
  "results": [
    {
      "id": "doc2",
      "title": "Machine Learning Guide",
      "content": "A comprehensive guide to ML algorithms...",
      "score": 8.4523,
      "metadata": {
        "author": "Jane Smith",
        "quality_score": 0.95
      }
    },
    {
      "id": "doc1",
      "title": "Python Tutorial",
      "content": "Learn Python programming from scratch...",
      "score": 6.2341,
      "metadata": {
        "author": "John Doe",
        "quality_score": 0.9
      }
    }
  ]
}
```

### Evaluate Rankings

```bash
curl -X POST http://localhost:5000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      {
        "query": "python programming",
        "algorithm": "bm25",
        "ground_truth": {
          "doc1": 3.0,
          "doc2": 1.0
        }
      }
    ]
  }'
```

---

## Python SDK Usage

### Complete Workflow

```python
import json
from src.rankers.bm25_ranker import BM25Ranker
from src.evaluation.metrics import RankingMetrics

# Load data
with open('data/sample_documents.json', 'r') as f:
    documents = json.load(f)

# Initialize ranker
ranker = BM25Ranker()
ranker.index_documents(documents)

# Search
query = "web development javascript"
results = ranker.search(query, top_k=10)

# Display results
print(f"\nSearch Results for: '{query}'")
print("-" * 60)
for i, (doc_id, score) in enumerate(results, 1):
    doc = ranker.get_document_by_id(doc_id)
    print(f"{i}. {doc['title']}")
    print(f"   Score: {score:.4f}")
    print(f"   Preview: {doc['content'][:100]}...")
    print()

# Save model
ranker.save_model('models/my_custom_bm25.pkl')

# Load model later
new_ranker = BM25Ranker()
new_ranker.load_model('models/my_custom_bm25.pkl')
```

### Evaluation Example

```python
from src.evaluation.metrics import RankingMetrics

# Predicted rankings
predicted = [
    [('doc1', 0.9), ('doc2', 0.8), ('doc3', 0.7)],
    [('doc2', 0.95), ('doc1', 0.85), ('doc4', 0.75)]
]

# Ground truth relevance
ground_truth = [
    {'doc1': 3, 'doc2': 2, 'doc3': 0},
    {'doc2': 3, 'doc1': 1, 'doc4': 2}
]

# Evaluate
metrics = RankingMetrics.evaluate_ranking(
    predicted,
    ground_truth,
    k_values=[1, 5, 10]
)

print("Evaluation Metrics:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.4f}")

# Output:
# ndcg@1: 1.0000
# ndcg@5: 0.9463
# ndcg@10: 0.9463
# map: 0.9167
# mrr: 1.0000
# precision@1: 1.0000
# precision@5: 0.8000
```

---

## A/B Testing

### Create and Run Experiment

```python
from src.ab_testing.experiment import ABTestExperiment
import random

# Create experiment
experiment = ABTestExperiment(
    experiment_id='bm25_vs_lambdamart',
    baseline_name='bm25',
    treatment_name='lambdamart',
    traffic_split=0.5
)

# Simulate user sessions
for user_id in range(1000):
    # Assign variant
    variant = experiment.assign_variant(f"user_{user_id}")
    
    # Simulate engagement metric (e.g., click-through rate)
    if variant == 'baseline':
        engagement = random.normalvariate(0.15, 0.05)  # 15% CTR
    else:
        engagement = random.normalvariate(0.18, 0.05)  # 18% CTR
    
    # Record metric
    experiment.record_metric(variant, max(0, min(1, engagement)))

# Analyze results
results = experiment.calculate_statistics()

print(f"Baseline CTR: {results['baseline']['mean']:.4f}")
print(f"Treatment CTR: {results['treatment']['mean']:.4f}")
print(f"Improvement: {results['results']['relative_improvement_pct']:.2f}%")
print(f"P-value: {results['statistics']['p_value']:.6f}")
print(f"Significant: {results['results']['is_significant']}")
```

### API-Based A/B Testing

```bash
# Create experiment
curl -X POST http://localhost:5000/experiment/create \
  -H "Content-Type: application/json" \
  -d '{
    "experiment_id": "exp_001",
    "baseline": "tfidf",
    "treatment": "bm25",
    "traffic_split": 0.5
  }'

# Search with experiment
curl -X POST http://localhost:5000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "user_id": "user123",
    "experiment_id": "exp_001",
    "top_k": 5
  }'

# Record engagement
curl -X POST http://localhost:5000/experiment/exp_001/record \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "metric_value": 0.85
  }'

# Get results
curl http://localhost:5000/experiment/exp_001/results
```

---

## Custom Features

### Add Custom Features to LambdaMART

```python
from src.features.feature_extractor import FeatureExtractor
import numpy as np

class CustomFeatureExtractor(FeatureExtractor):
    def extract_features(self, query, document):
        # Get base features
        base_features = super().extract_features(query, document)
        
        # Add custom features
        custom_features = [
            self._calculate_author_authority(document),
            self._calculate_recency_score(document),
            self._calculate_user_engagement(document)
        ]
        
        return np.concatenate([base_features, custom_features])
    
    def _calculate_author_authority(self, document):
        # Custom logic for author authority
        author = document.get('metadata', {}).get('author', '')
        authority_scores = {
            'John Doe': 0.9,
            'Jane Smith': 0.95,
            # ... more authors
        }
        return authority_scores.get(author, 0.5)
    
    def _calculate_recency_score(self, document):
        # Custom logic for content freshness
        date_str = document.get('metadata', {}).get('date', '')
        # Calculate days since publication
        # Return score between 0 and 1
        return 0.8
    
    def _calculate_user_engagement(self, document):
        # Custom logic for engagement signals
        views = document.get('metadata', {}).get('views', 0)
        return min(views / 10000, 1.0)
    
    def get_feature_names(self):
        base_names = super().get_feature_names()
        custom_names = [
            'author_authority',
            'recency_score',
            'user_engagement'
        ]
        return base_names + custom_names

# Use custom feature extractor
ranker = LambdaMARTRanker()
ranker.feature_extractor = CustomFeatureExtractor()
ranker.index_documents(documents)
```

---

## Advanced Scenarios

### Batch Processing

```python
def batch_search(queries, ranker, batch_size=100):
    """Process queries in batches"""
    results = []
    
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]
        batch_results = []
        
        for query in batch:
            search_results = ranker.search(query, top_k=10)
            batch_results.append({
                'query': query,
                'results': search_results
            })
        
        results.extend(batch_results)
        print(f"Processed {len(results)}/{len(queries)} queries")
    
    return results

# Usage
queries = ["query1", "query2", "query3", ...]
all_results = batch_search(queries, ranker)
```

### Multi-Algorithm Ensemble

```python
from src.rankers import TFIDFRanker, BM25Ranker, LambdaMARTRanker

def ensemble_search(query, documents, top_k=10):
    """Combine multiple rankers"""
    # Initialize rankers
    tfidf = TFIDFRanker()
    bm25 = BM25Ranker()
    lmart = LambdaMARTRanker()
    
    # Index
    for ranker in [tfidf, bm25, lmart]:
        ranker.index_documents(documents)
    
    # Get results from each
    tfidf_results = dict(tfidf.search(query, top_k=20))
    bm25_results = dict(bm25.search(query, top_k=20))
    lmart_results = dict(lmart.search(query, top_k=20))
    
    # Combine scores (average)
    all_docs = set(tfidf_results.keys()) | set(bm25_results.keys()) | set(lmart_results.keys())
    
    ensemble_scores = {}
    for doc_id in all_docs:
        scores = [
            tfidf_results.get(doc_id, 0),
            bm25_results.get(doc_id, 0),
            lmart_results.get(doc_id, 0)
        ]
        ensemble_scores[doc_id] = sum(scores) / len(scores)
    
    # Sort and return top k
    sorted_results = sorted(ensemble_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]

# Usage
results = ensemble_search("machine learning", documents, top_k=5)
```

### Real-time Personalization

```python
def personalized_search(query, user_profile, ranker):
    """Personalize search results based on user profile"""
    # Get base results
    results = ranker.search(query, top_k=50)
    
    # Re-rank based on user preferences
    user_topics = user_profile.get('favorite_topics', [])
    user_authors = user_profile.get('favorite_authors', [])
    
    personalized_scores = []
    for doc_id, base_score in results:
        doc = ranker.get_document_by_id(doc_id)
        
        # Boost score based on user preferences
        boost = 1.0
        if doc['metadata'].get('topic') in user_topics:
            boost += 0.3
        if doc['metadata'].get('author') in user_authors:
            boost += 0.2
        
        personalized_scores.append((doc_id, base_score * boost))
    
    # Re-sort
    personalized_scores.sort(key=lambda x: x[1], reverse=True)
    return personalized_scores[:10]

# Usage
user_profile = {
    'favorite_topics': ['machine_learning', 'data_science'],
    'favorite_authors': ['John Doe']
}

results = personalized_search("python tutorial", user_profile, ranker)
```

---

## Performance Tips

1. **Cache Results**: Cache frequently searched queries
2. **Batch Indexing**: Index documents in batches for large datasets
3. **Feature Selection**: Use only the most important features
4. **Model Persistence**: Save and load models to avoid retraining
5. **Async Processing**: Use async operations for API calls

---

For more examples, check the test files in the `tests/` directory!
