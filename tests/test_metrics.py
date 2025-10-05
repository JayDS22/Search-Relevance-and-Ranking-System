"""
Unit tests for evaluation metrics
"""
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.metrics import RankingMetrics


class TestNDCG:
    """Test nDCG metric"""
    
    def test_perfect_ranking(self):
        """Test nDCG with perfect ranking"""
        relevance = [3, 2, 1, 0, 0]
        ndcg = RankingMetrics.ndcg_at_k(relevance, k=5)
        assert ndcg == pytest.approx(1.0)
    
    def test_worst_ranking(self):
        """Test nDCG with worst ranking"""
        relevance = [0, 0, 0, 3, 2]
        ndcg = RankingMetrics.ndcg_at_k(relevance, k=5)
        assert ndcg < 0.5
    
    def test_partial_ranking(self):
        """Test nDCG with partial ranking"""
        relevance = [2, 3, 1, 0, 0]
        ndcg = RankingMetrics.ndcg_at_k(relevance, k=5)
        assert 0.9 < ndcg < 1.0
    
    def test_empty_relevance(self):
        """Test nDCG with empty list"""
        ndcg = RankingMetrics.ndcg_at_k([], k=10)
        assert ndcg == 0.0
    
    def test_ndcg_at_different_k(self):
        """Test nDCG at different cutoffs"""
        relevance = [3, 2, 1, 1, 0, 0]
        
        ndcg_at_3 = RankingMetrics.ndcg_at_k(relevance, k=3)
        ndcg_at_5 = RankingMetrics.ndcg_at_k(relevance, k=5)
        
        assert ndcg_at_3 > 0
        assert ndcg_at_5 > 0


class TestAveragePrecision:
    """Test Average Precision metric"""
    
    def test_perfect_ranking(self):
        """Test AP with all relevant docs first"""
        relevance = [1, 1, 1, 0, 0]
        ap = RankingMetrics.average_precision(relevance)
        assert ap == 1.0
    
    def test_no_relevant(self):
        """Test AP with no relevant documents"""
        relevance = [0, 0, 0, 0, 0]
        ap = RankingMetrics.average_precision(relevance)
        assert ap == 0.0
    
    def test_mixed_ranking(self):
        """Test AP with mixed relevant/irrelevant"""
        relevance = [1, 0, 1, 0, 1]
        ap = RankingMetrics.average_precision(relevance)
        assert 0.5 < ap < 1.0


class TestPrecisionRecall:
    """Test Precision and Recall metrics"""
    
    def test_precision_at_k(self):
        """Test precision at k"""
        relevance = [1, 1, 0, 1, 0]
        
        p_at_3 = RankingMetrics.precision_at_k(relevance, k=3)
        assert p_at_3 == pytest.approx(2/3)
        
        p_at_5 = RankingMetrics.precision_at_k(relevance, k=5)
        assert p_at_5 == pytest.approx(3/5)
    
    def test_recall_at_k(self):
        """Test recall at k"""
        relevance = [1, 1, 0, 1, 0, 1, 0]
        
        # 4 total relevant, 2 in top 3
        r_at_3 = RankingMetrics.recall_at_k(relevance, k=3)
        assert r_at_3 == pytest.approx(2/4)
        
        # 4 total relevant, 3 in top 5
        r_at_5 = RankingMetrics.recall_at_k(relevance, k=5)
        assert r_at_5 == pytest.approx(3/4)
    
    def test_precision_no_relevant(self):
        """Test precision with no relevant docs"""
        relevance = [0, 0, 0, 0, 0]
        p = RankingMetrics.precision_at_k(relevance, k=5)
        assert p == 0.0
    
    def test_recall_no_relevant(self):
        """Test recall with no relevant docs"""
        relevance = [0, 0, 0, 0, 0]
        r = RankingMetrics.recall_at_k(relevance, k=5)
        assert r == 0.0


class TestReciprocalRank:
    """Test Reciprocal Rank metrics"""
    
    def test_first_position(self):
        """Test RR when first doc is relevant"""
        relevance = [1, 0, 0, 0]
        rr = RankingMetrics.reciprocal_rank(relevance)
        assert rr == 1.0
    
    def test_second_position(self):
        """Test RR when second doc is relevant"""
        relevance = [0, 1, 0, 0]
        rr = RankingMetrics.reciprocal_rank(relevance)
        assert rr == 0.5
    
    def test_third_position(self):
        """Test RR when third doc is relevant"""
        relevance = [0, 0, 1, 0]
        rr = RankingMetrics.reciprocal_rank(relevance)
        assert rr == pytest.approx(1/3)
    
    def test_no_relevant(self):
        """Test RR with no relevant docs"""
        relevance = [0, 0, 0, 0]
        rr = RankingMetrics.reciprocal_rank(relevance)
        assert rr == 0.0
    
    def test_mrr(self):
        """Test Mean Reciprocal Rank"""
        all_relevance = [
            [1, 0, 0],  # RR = 1.0
            [0, 1, 0],  # RR = 0.5
            [0, 0, 1],  # RR = 1/3
        ]
        
        mrr = RankingMetrics.mean_reciprocal_rank(all_relevance)
        expected = (1.0 + 0.5 + 1/3) / 3
        assert mrr == pytest.approx(expected)


class TestEvaluateRanking:
    """Test comprehensive ranking evaluation"""
    
    def test_evaluate_ranking(self):
        """Test full evaluation pipeline"""
        predicted_rankings = [
            [('doc1', 0.9), ('doc2', 0.8), ('doc3', 0.7)],
            [('doc2', 0.95), ('doc1', 0.85), ('doc4', 0.75)]
        ]
        
        ground_truth = [
            {'doc1': 3, 'doc2': 2, 'doc3': 0},
            {'doc2': 3, 'doc1': 1, 'doc4': 0}
        ]
        
        metrics = RankingMetrics.evaluate_ranking(
            predicted_rankings,
            ground_truth,
            k_values=[1, 3]
        )
        
        # Check that all expected metrics are present
        assert 'ndcg@1' in metrics
        assert 'ndcg@3' in metrics
        assert 'map' in metrics
        assert 'mrr' in metrics
        assert 'precision@1' in metrics
        assert 'recall@1' in metrics
        
        # Check that all values are valid
        for value in metrics.values():
            assert 0 <= value <= 1


class TestCompareRankings:
    """Test ranking comparison"""
    
    def test_compare_rankings(self):
        """Test comparison of two ranking systems"""
        baseline = {
            'ndcg@10': 0.75,
            'map': 0.70,
            'precision@5': 0.80
        }
        
        treatment = {
            'ndcg@10': 0.85,
            'map': 0.78,
            'precision@5': 0.88
        }
        
        comparison = RankingMetrics.compare_rankings(baseline, treatment)
        
        assert 'ndcg@10' in comparison
        assert comparison['ndcg@10']['baseline'] == 0.75
        assert comparison['ndcg@10']['treatment'] == 0.85
        assert comparison['ndcg@10']['absolute_improvement'] == pytest.approx(0.10)
        assert comparison['ndcg@10']['relative_improvement_pct'] > 0
