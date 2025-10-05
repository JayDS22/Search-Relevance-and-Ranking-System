"""
Evaluation metrics for ranking quality
"""
import numpy as np
from typing import List, Dict, Tuple
import config


class RankingMetrics:
    """Calculate ranking evaluation metrics"""
    
    @staticmethod
    def ndcg_at_k(relevance_scores: List[float], k: int = 10) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at k
        
        Args:
            relevance_scores: List of relevance scores (higher is better)
            k: Position cutoff
            
        Returns:
            nDCG@k score (0-1)
        """
        relevance_scores = np.array(relevance_scores[:k])
        
        if len(relevance_scores) == 0:
            return 0.0
        
        # DCG
        dcg = relevance_scores[0]
        for i in range(1, len(relevance_scores)):
            dcg += relevance_scores[i] / np.log2(i + 2)
        
        # IDCG (ideal DCG with perfect ranking)
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = ideal_scores[0]
        for i in range(1, len(ideal_scores)):
            idcg += ideal_scores[i] / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def average_precision(relevance_scores: List[float], threshold: float = 1.0) -> float:
        """
        Calculate Average Precision
        
        Args:
            relevance_scores: List of relevance scores
            threshold: Minimum score to consider as relevant
            
        Returns:
            Average Precision score
        """
        relevant = [1 if score >= threshold else 0 for score in relevance_scores]
        
        if sum(relevant) == 0:
            return 0.0
        
        precision_sum = 0.0
        num_relevant = 0
        
        for i, rel in enumerate(relevant):
            if rel == 1:
                num_relevant += 1
                precision_at_i = num_relevant / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / sum(relevant)
    
    @staticmethod
    def mean_average_precision(all_relevance_scores: List[List[float]], threshold: float = 1.0) -> float:
        """
        Calculate Mean Average Precision across multiple queries
        
        Args:
            all_relevance_scores: List of relevance score lists
            threshold: Minimum score to consider as relevant
            
        Returns:
            MAP score
        """
        if not all_relevance_scores:
            return 0.0
        
        ap_scores = [
            RankingMetrics.average_precision(scores, threshold)
            for scores in all_relevance_scores
        ]
        
        return np.mean(ap_scores)
    
    @staticmethod
    def precision_at_k(relevance_scores: List[float], k: int, threshold: float = 1.0) -> float:
        """
        Calculate Precision at k
        
        Args:
            relevance_scores: List of relevance scores
            k: Position cutoff
            threshold: Minimum score to consider as relevant
            
        Returns:
            Precision@k score
        """
        if k == 0 or len(relevance_scores) == 0:
            return 0.0
        
        top_k = relevance_scores[:k]
        relevant = sum(1 for score in top_k if score >= threshold)
        
        return relevant / k
    
    @staticmethod
    def recall_at_k(relevance_scores: List[float], k: int, threshold: float = 1.0) -> float:
        """
        Calculate Recall at k
        
        Args:
            relevance_scores: List of relevance scores
            k: Position cutoff
            threshold: Minimum score to consider as relevant
            
        Returns:
            Recall@k score
        """
        total_relevant = sum(1 for score in relevance_scores if score >= threshold)
        
        if total_relevant == 0:
            return 0.0
        
        top_k = relevance_scores[:k]
        relevant_in_top_k = sum(1 for score in top_k if score >= threshold)
        
        return relevant_in_top_k / total_relevant
    
    @staticmethod
    def reciprocal_rank(relevance_scores: List[float], threshold: float = 1.0) -> float:
        """
        Calculate Reciprocal Rank (RR)
        
        Args:
            relevance_scores: List of relevance scores
            threshold: Minimum score to consider as relevant
            
        Returns:
            Reciprocal rank
        """
        for i, score in enumerate(relevance_scores):
            if score >= threshold:
                return 1.0 / (i + 1)
        
        return 0.0
    
    @staticmethod
    def mean_reciprocal_rank(all_relevance_scores: List[List[float]], threshold: float = 1.0) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR) across multiple queries
        
        Args:
            all_relevance_scores: List of relevance score lists
            threshold: Minimum score to consider as relevant
            
        Returns:
            MRR score
        """
        if not all_relevance_scores:
            return 0.0
        
        rr_scores = [
            RankingMetrics.reciprocal_rank(scores, threshold)
            for scores in all_relevance_scores
        ]
        
        return np.mean(rr_scores)
    
    @staticmethod
    def evaluate_ranking(
        predicted_rankings: List[List[Tuple[str, float]]],
        ground_truth: List[Dict[str, float]],
        k_values: List[int] = None
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of ranking quality
        
        Args:
            predicted_rankings: List of ranked results per query [(doc_id, score), ...]
            ground_truth: List of relevance judgments per query {doc_id: relevance, ...}
            k_values: List of k values for metrics (default from config)
            
        Returns:
            Dictionary of metric scores
        """
        if k_values is None:
            k_values = config.EVAL_TOP_K
        
        # Extract relevance scores in predicted order
        all_relevance_scores = []
        for ranking, truth in zip(predicted_rankings, ground_truth):
            scores = [truth.get(doc_id, 0.0) for doc_id, _ in ranking]
            all_relevance_scores.append(scores)
        
        # Calculate metrics
        metrics = {}
        
        # nDCG at different k values
        for k in k_values:
            ndcg_scores = [
                RankingMetrics.ndcg_at_k(scores, k)
                for scores in all_relevance_scores
            ]
            metrics[f'ndcg@{k}'] = np.mean(ndcg_scores)
        
        # MAP
        metrics['map'] = RankingMetrics.mean_average_precision(all_relevance_scores)
        
        # MRR
        metrics['mrr'] = RankingMetrics.mean_reciprocal_rank(all_relevance_scores)
        
        # Precision and Recall at different k values
        for k in k_values:
            precision_scores = [
                RankingMetrics.precision_at_k(scores, k)
                for scores in all_relevance_scores
            ]
            recall_scores = [
                RankingMetrics.recall_at_k(scores, k)
                for scores in all_relevance_scores
            ]
            
            metrics[f'precision@{k}'] = np.mean(precision_scores)
            metrics[f'recall@{k}'] = np.mean(recall_scores)
        
        return metrics
    
    @staticmethod
    def compare_rankings(
        baseline_metrics: Dict[str, float],
        treatment_metrics: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare two ranking systems
        
        Args:
            baseline_metrics: Metrics from baseline system
            treatment_metrics: Metrics from treatment system
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {}
        
        for metric_name in baseline_metrics:
            if metric_name in treatment_metrics:
                baseline_val = baseline_metrics[metric_name]
                treatment_val = treatment_metrics[metric_name]
                
                # Calculate absolute and relative improvements
                abs_improvement = treatment_val - baseline_val
                rel_improvement = (abs_improvement / baseline_val * 100) if baseline_val != 0 else 0
                
                comparison[metric_name] = {
                    'baseline': baseline_val,
                    'treatment': treatment_val,
                    'absolute_improvement': abs_improvement,
                    'relative_improvement_pct': rel_improvement
                }
        
        return comparison
