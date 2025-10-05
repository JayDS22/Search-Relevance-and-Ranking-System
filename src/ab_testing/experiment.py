"""
A/B Testing framework for comparing ranking algorithms
"""
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import random
import json
from datetime import datetime
import config


class ABTestExperiment:
    """A/B testing framework for ranking algorithms"""
    
    def __init__(
        self,
        experiment_id: str,
        baseline_name: str,
        treatment_name: str,
        traffic_split: float = None
    ):
        """
        Initialize A/B test experiment
        
        Args:
            experiment_id: Unique identifier for the experiment
            baseline_name: Name of baseline algorithm
            treatment_name: Name of treatment algorithm
            traffic_split: Fraction of traffic to treatment (default from config)
        """
        self.experiment_id = experiment_id
        self.baseline_name = baseline_name
        self.treatment_name = treatment_name
        self.traffic_split = traffic_split if traffic_split is not None else config.AB_TEST_TRAFFIC_SPLIT
        
        self.baseline_metrics = []
        self.treatment_metrics = []
        self.user_assignments = {}
        
    def assign_variant(self, user_id: str) -> str:
        """
        Assign user to control or treatment group
        
        Args:
            user_id: User identifier
            
        Returns:
            Variant name ('baseline' or 'treatment')
        """
        if user_id in self.user_assignments:
            return self.user_assignments[user_id]
        
        # Consistent assignment using hash
        hash_val = hash(f"{self.experiment_id}_{user_id}")
        is_treatment = (hash_val % 100) / 100 < self.traffic_split
        
        variant = 'treatment' if is_treatment else 'baseline'
        self.user_assignments[user_id] = variant
        
        return variant
    
    def record_metric(self, variant: str, metric_value: float) -> None:
        """
        Record a metric observation
        
        Args:
            variant: 'baseline' or 'treatment'
            metric_value: Observed metric value
        """
        if variant == 'baseline':
            self.baseline_metrics.append(metric_value)
        elif variant == 'treatment':
            self.treatment_metrics.append(metric_value)
    
    def calculate_statistics(self) -> Dict:
        """
        Calculate statistical test results
        
        Returns:
            Dictionary with statistical results
        """
        if len(self.baseline_metrics) < 2 or len(self.treatment_metrics) < 2:
            return {
                'error': 'Insufficient data for statistical testing',
                'baseline_n': len(self.baseline_metrics),
                'treatment_n': len(self.treatment_metrics)
            }
        
        baseline = np.array(self.baseline_metrics)
        treatment = np.array(self.treatment_metrics)
        
        # Calculate means and standard deviations
        baseline_mean = np.mean(baseline)
        treatment_mean = np.mean(treatment)
        baseline_std = np.std(baseline, ddof=1)
        treatment_std = np.std(treatment, ddof=1)
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(treatment, baseline)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(baseline) - 1) * baseline_std**2 + 
                              (len(treatment) - 1) * treatment_std**2) / 
                             (len(baseline) + len(treatment) - 2))
        cohens_d = (treatment_mean - baseline_mean) / pooled_std if pooled_std > 0 else 0
        
        # Calculate confidence interval
        se_diff = np.sqrt(baseline_std**2 / len(baseline) + treatment_std**2 / len(treatment))
        ci_95 = stats.t.interval(
            0.95,
            len(baseline) + len(treatment) - 2,
            loc=treatment_mean - baseline_mean,
            scale=se_diff
        )
        
        # Calculate relative improvement
        rel_improvement = ((treatment_mean - baseline_mean) / baseline_mean * 100) if baseline_mean != 0 else 0
        
        results = {
            'experiment_id': self.experiment_id,
            'baseline': {
                'name': self.baseline_name,
                'n': len(baseline),
                'mean': float(baseline_mean),
                'std': float(baseline_std),
                'sem': float(baseline_std / np.sqrt(len(baseline)))
            },
            'treatment': {
                'name': self.treatment_name,
                'n': len(treatment),
                'mean': float(treatment_mean),
                'std': float(treatment_std),
                'sem': float(treatment_std / np.sqrt(len(treatment)))
            },
            'statistics': {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'cohens_d': float(cohens_d),
                'ci_95_lower': float(ci_95[0]),
                'ci_95_upper': float(ci_95[1])
            },
            'results': {
                'absolute_improvement': float(treatment_mean - baseline_mean),
                'relative_improvement_pct': float(rel_improvement),
                'is_significant': p_value < (1 - config.AB_TEST_CONFIDENCE_LEVEL),
                'winner': self.treatment_name if (p_value < (1 - config.AB_TEST_CONFIDENCE_LEVEL) and treatment_mean > baseline_mean) else 'inconclusive'
            }
        }
        
        return results
    
    def required_sample_size(
        self,
        baseline_mean: float,
        baseline_std: float,
        mde: float,
        alpha: float = 0.05,
        power: float = 0.8
    ) -> int:
        """
        Calculate required sample size per variant
        
        Args:
            baseline_mean: Expected baseline mean
            baseline_std: Expected baseline standard deviation
            mde: Minimum detectable effect (absolute)
            alpha: Significance level
            power: Statistical power
            
        Returns:
            Required sample size per variant
        """
        effect_size = mde / baseline_std if baseline_std > 0 else 0
        
        if effect_size == 0:
            return float('inf')
        
        # Using formula for two-sample t-test
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        
        return int(np.ceil(n))
    
    def is_ready_for_analysis(self) -> Tuple[bool, str]:
        """
        Check if experiment has sufficient data
        
        Returns:
            Tuple of (ready, message)
        """
        min_size = config.AB_TEST_MIN_SAMPLE_SIZE
        
        baseline_n = len(self.baseline_metrics)
        treatment_n = len(self.treatment_metrics)
        
        if baseline_n < min_size:
            return False, f"Baseline has {baseline_n} samples, need {min_size}"
        
        if treatment_n < min_size:
            return False, f"Treatment has {treatment_n} samples, need {min_size}"
        
        return True, "Experiment has sufficient data"
    
    def save_results(self, filepath: str = None) -> None:
        """Save experiment results to file"""
        if filepath is None:
            filepath = f"experiment_{self.experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        results = self.calculate_statistics()
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    
    def load_results(self, filepath: str) -> Dict:
        """Load experiment results from file"""
        with open(filepath, 'r') as f:
            return json.load(f)


class ABTestManager:
    """Manage multiple A/B test experiments"""
    
    def __init__(self):
        self.experiments = {}
    
    def create_experiment(
        self,
        experiment_id: str,
        baseline_name: str,
        treatment_name: str,
        traffic_split: float = None
    ) -> ABTestExperiment:
        """Create a new experiment"""
        experiment = ABTestExperiment(
            experiment_id,
            baseline_name,
            treatment_name,
            traffic_split
        )
        self.experiments[experiment_id] = experiment
        return experiment
    
    def get_experiment(self, experiment_id: str) -> Optional[ABTestExperiment]:
        """Get an existing experiment"""
        return self.experiments.get(experiment_id)
    
    def list_experiments(self) -> List[str]:
        """List all experiment IDs"""
        return list(self.experiments.keys())
    
    def compare_multiple_variants(
        self,
        metrics_by_variant: Dict[str, List[float]]
    ) -> Dict:
        """
        Compare multiple variants using ANOVA
        
        Args:
            metrics_by_variant: Dictionary mapping variant names to metric lists
            
        Returns:
            ANOVA results
        """
        variant_names = list(metrics_by_variant.keys())
        variant_data = [metrics_by_variant[name] for name in variant_names]
        
        # Perform one-way ANOVA
        f_stat, p_value = stats.f_oneway(*variant_data)
        
        # Calculate means
        means = {name: np.mean(data) for name, data in metrics_by_variant.items()}
        
        results = {
            'variants': variant_names,
            'means': means,
            'f_statistic': float(f_stat),
            'p_value': float(p_value),
            'is_significant': p_value < 0.05,
            'best_variant': max(means.items(), key=lambda x: x[1])[0]
        }
        
        return results
