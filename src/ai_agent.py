"""
AI Agent for RAG System Optimization
Auto-monitors metrics and optimizes system parameters
"""

import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import statistics


class OptimizationAction(Enum):
    """Actions the agent can take"""
    ADJUST_TOP_K = "adjust_top_k"
    ADJUST_ALPHA = "adjust_alpha"
    CHANGE_RETRIEVAL_METHOD = "change_retrieval_method"
    ADJUST_TEMPERATURE = "adjust_temperature"
    ADD_SYSTEM_PROMPT = "add_system_prompt"
    ADJUST_MAX_TOKENS = "adjust_max_tokens"
    ENABLE_HYBRID = "enable_hybrid"
    INCREASE_CONTEXT = "increase_context"
    NO_ACTION = "no_action"


@dataclass
class SystemConfig:
    """Current system configuration"""
    top_k: int = 5
    hybrid_alpha: float = 0.7
    retrieval_method: str = "hybrid"
    temperature: float = 0.5
    max_tokens: int = 1000
    enable_evaluation: bool = True
    use_llm_judge: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'top_k': self.top_k,
            'hybrid_alpha': self.hybrid_alpha,
            'retrieval_method': self.retrieval_method,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'enable_evaluation': self.enable_evaluation,
            'use_llm_judge': self.use_llm_judge
        }


@dataclass
class MetricsSnapshot:
    """Snapshot of current metrics"""
    timestamp: datetime
    faithfulness: float
    relevance: float
    hallucination_rate: float
    correctness: Optional[float] = None
    recall_at_5: Optional[float] = None
    precision_at_5: Optional[float] = None
    mrr: Optional[float] = None
    latency_ms: float = 0
    tokens: int = 0
    
    def overall_score(self) -> float:
        """Calculate overall quality score"""
        scores = [
            self.faithfulness,
            self.relevance,
            1 - self.hallucination_rate  # Lower is better
        ]
        if self.correctness is not None:
            scores.append(self.correctness)
        if self.recall_at_5 is not None:
            scores.append(self.recall_at_5)
        
        return sum(scores) / len(scores) if scores else 0


class MetricsAnalyzer:
    """Analyze metrics trends and identify issues"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.metrics_history: List[MetricsSnapshot] = []
    
    def add_snapshot(self, snapshot: MetricsSnapshot):
        """Add metrics snapshot"""
        self.metrics_history.append(snapshot)
        # Keep only recent snapshots
        if len(self.metrics_history) > self.window_size:
            self.metrics_history = self.metrics_history[-self.window_size:]
    
    def get_trend(self, metric_name: str) -> Tuple[str, float]:
        """
        Analyze trend for a metric
        
        Returns: (trend_direction, change_percent)
        """
        if len(self.metrics_history) < 2:
            return "insufficient_data", 0.0
        
        values = [getattr(m, metric_name) for m in self.metrics_history]
        
        if len(values) < 2:
            return "insufficient_data", 0.0
        
        # Calculate trend
        recent_avg = statistics.mean(values[-3:]) if len(values) >= 3 else values[-1]
        old_avg = statistics.mean(values[:3]) if len(values) >= 3 else values[0]
        
        if old_avg == 0:
            change_percent = 0.0
        else:
            change_percent = ((recent_avg - old_avg) / old_avg) * 100
        
        if change_percent > 5:
            return "improving", change_percent
        elif change_percent < -5:
            return "degrading", change_percent
        else:
            return "stable", change_percent
    
    def identify_issues(self, config: SystemConfig) -> List[str]:
        """Identify problems with current metrics"""
        if not self.metrics_history:
            return []
        
        latest = self.metrics_history[-1]
        issues = []
        
        # Check faithfulness
        if latest.faithfulness < 0.75:
            issues.append("low_faithfulness")
        
        # Check relevance
        if latest.relevance < 0.70:
            issues.append("low_relevance")
        
        # Check hallucination
        if latest.hallucination_rate > 0.15:
            issues.append("high_hallucination")
        
        # Check latency
        if latest.latency_ms > 2000:
            issues.append("high_latency")
        
        # Check token usage
        if latest.tokens > 3000:
            issues.append("high_token_usage")
        
        # Check retrieval quality
        if latest.recall_at_5 is not None and latest.recall_at_5 < 0.70:
            issues.append("low_recall")
        
        if latest.precision_at_5 is not None and latest.precision_at_5 < 0.50:
            issues.append("low_precision")
        
        return issues


class ActionPlanner:
    """Plan optimization actions based on issues"""
    
    # Action recommendations by issue
    ACTION_MAPPING = {
        'low_faithfulness': [
            (OptimizationAction.ADJUST_TEMPERATURE, {'temperature': 0.3}),
            (OptimizationAction.ADD_SYSTEM_PROMPT, {'emphasis': 'faithfulness'}),
            (OptimizationAction.INCREASE_CONTEXT, {'top_k': 7}),
        ],
        'low_relevance': [
            (OptimizationAction.ADJUST_ALPHA, {'alpha': 0.5}),
            (OptimizationAction.INCREASE_CONTEXT, {'top_k': 5}),
            (OptimizationAction.CHANGE_RETRIEVAL_METHOD, {'method': 'hybrid'}),
        ],
        'high_hallucination': [
            (OptimizationAction.ADJUST_TEMPERATURE, {'temperature': 0.2}),
            (OptimizationAction.ADD_SYSTEM_PROMPT, {'emphasis': 'context_only'}),
            (OptimizationAction.ADJUST_MAX_TOKENS, {'max_tokens': 800}),
        ],
        'high_latency': [
            (OptimizationAction.ADJUST_TOP_K, {'top_k': 3}),
            (OptimizationAction.ADJUST_MAX_TOKENS, {'max_tokens': 800}),
            (OptimizationAction.CHANGE_RETRIEVAL_METHOD, {'method': 'semantic'}),
        ],
        'high_token_usage': [
            (OptimizationAction.ADJUST_MAX_TOKENS, {'max_tokens': 800}),
            (OptimizationAction.ADJUST_TOP_K, {'top_k': 3}),
        ],
        'low_recall': [
            (OptimizationAction.INCREASE_CONTEXT, {'top_k': 7}),
            (OptimizationAction.ADJUST_ALPHA, {'alpha': 1.0}),
            (OptimizationAction.CHANGE_RETRIEVAL_METHOD, {'method': 'semantic'}),
        ],
        'low_precision': [
            (OptimizationAction.ADJUST_ALPHA, {'alpha': 0.0}),
            (OptimizationAction.ADJUST_TOP_K, {'top_k': 3}),
            (OptimizationAction.CHANGE_RETRIEVAL_METHOD, {'method': 'keyword'}),
        ],
    }
    
    @staticmethod
    def plan_actions(
        issues: List[str],
        config: SystemConfig,
        analyzer: MetricsAnalyzer
    ) -> List[Tuple[OptimizationAction, Dict]]:
        """
        Plan optimization actions
        
        Returns: List of (action, parameters)
        """
        if not issues:
            return [(OptimizationAction.NO_ACTION, {})]
        
        actions = []
        seen_actions = set()
        
        # Primary issue (most important)
        primary_issue = issues[0]
        
        if primary_issue in ActionPlanner.ACTION_MAPPING:
            recommended = ActionPlanner.ACTION_MAPPING[primary_issue]
            for action, params in recommended[:2]:  # Top 2 recommendations
                if action not in seen_actions:
                    actions.append((action, params))
                    seen_actions.add(action)
        
        return actions if actions else [(OptimizationAction.NO_ACTION, {})]


class RAGOptimizationAgent:
    """
    AI Agent for RAG System Optimization
    Monitors metrics and automatically optimizes system
    """
    
    def __init__(self, rag_engine, llm_client=None, history_file: str = "agent_history.jsonl"):
        """
        Initialize optimization agent
        
        Args:
            rag_engine: RAGEngine instance to optimize
            llm_client: Optional LLM for reasoning
            history_file: Path to save agent actions history
        """
        self.rag_engine = rag_engine
        self.llm_client = llm_client
        self.history_file = history_file
        
        self.config = SystemConfig(
            top_k=rag_engine.top_k,
            hybrid_alpha=rag_engine.hybrid_alpha,
            retrieval_method="hybrid"
        )
        
        self.analyzer = MetricsAnalyzer(window_size=10)
        self.action_history: List[Dict] = []
        
        # Performance thresholds
        self.thresholds = {
            'faithfulness': 0.85,
            'relevance': 0.80,
            'hallucination_rate': 0.10,
            'latency_ms': 1500,
        }
    
    def observe(self, metrics: MetricsSnapshot) -> None:
        """
        Observe current metrics
        
        Args:
            metrics: Current metrics snapshot
        """
        self.analyzer.add_snapshot(metrics)
    
    def analyze_and_plan(self) -> List[Tuple[OptimizationAction, Dict]]:
        """
        Analyze current state and plan optimizations
        
        Returns: List of planned actions
        """
        issues = self.analyzer.identify_issues(self.config)
        actions = ActionPlanner.plan_actions(issues, self.config, self.analyzer)
        
        return actions
    
    def execute_action(
        self,
        action: OptimizationAction,
        parameters: Dict
    ) -> bool:
        """
        Execute optimization action
        
        Args:
            action: Action to execute
            parameters: Action parameters
            
        Returns: Success flag
        """
        try:
            if action == OptimizationAction.ADJUST_TOP_K:
                new_k = parameters.get('top_k', self.config.top_k)
                self.rag_engine.top_k = new_k
                self.config.top_k = new_k
                
            elif action == OptimizationAction.ADJUST_ALPHA:
                new_alpha = parameters.get('alpha', self.config.hybrid_alpha)
                self.rag_engine.hybrid_alpha = new_alpha
                self.config.hybrid_alpha = new_alpha
                
            elif action == OptimizationAction.CHANGE_RETRIEVAL_METHOD:
                method = parameters.get('method', 'hybrid')
                self.config.retrieval_method = method
                
            elif action == OptimizationAction.ADJUST_TEMPERATURE:
                temp = parameters.get('temperature', 0.7)
                self.rag_engine.llm_client.temperature = temp
                self.config.temperature = temp
                
            elif action == OptimizationAction.ADJUST_MAX_TOKENS:
                tokens = parameters.get('max_tokens', 1000)
                self.rag_engine.llm_client.max_tokens = tokens
                self.config.max_tokens = tokens
                
            elif action == OptimizationAction.INCREASE_CONTEXT:
                top_k = parameters.get('top_k', self.config.top_k + 2)
                self.rag_engine.top_k = top_k
                self.config.top_k = top_k
                
            elif action == OptimizationAction.ADD_SYSTEM_PROMPT:
                # This would require modifying LLM client system prompt
                emphasis = parameters.get('emphasis', 'default')
                # Implementation depends on LLM client structure
                
            elif action == OptimizationAction.NO_ACTION:
                return True
            
            # Log action
            self._log_action(action, parameters, success=True)
            return True
            
        except Exception as e:
            print(f"Error executing action {action}: {str(e)}")
            self._log_action(action, parameters, success=False, error=str(e))
            return False
    
    def optimize(self, test_query: str, max_iterations: int = 3) -> Dict:
        """
        Run optimization loop
        
        Args:
            test_query: Query to use for testing
            max_iterations: Max optimization iterations
            
        Returns: Optimization report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'iterations': [],
            'initial_config': self.config.to_dict(),
            'final_config': None,
            'improvement': 0.0,
        }
        
        # Get initial metrics
        initial_result = self.rag_engine.query(test_query)
        initial_metrics = self._extract_metrics(initial_result)
        self.observe(initial_metrics)
        initial_score = initial_metrics.overall_score()
        
        iteration_data = {
            'iteration': 0,
            'config': self.config.to_dict(),
            'metrics': self._metrics_to_dict(initial_metrics),
            'score': initial_score,
            'action': None
        }
        report['iterations'].append(iteration_data)
        
        print(f"\n[Agent] Initial score: {initial_score:.2%}")
        
        # Optimization loop
        for iteration in range(1, max_iterations + 1):
            print(f"\n[Agent] Iteration {iteration}/{max_iterations}")
            
            # Plan actions
            actions = self.analyze_and_plan()
            
            if not actions or actions[0][0] == OptimizationAction.NO_ACTION:
                print("[Agent] No improvements needed")
                break
            
            # Execute first action
            action, params = actions[0]
            print(f"[Agent] Executing: {action.value} with {params}")
            
            if self.execute_action(action, params):
                # Test new configuration
                result = self.rag_engine.query(test_query)
                metrics = self._extract_metrics(result)
                self.observe(metrics)
                
                score = metrics.overall_score()
                improvement = score - initial_score
                
                iteration_data = {
                    'iteration': iteration,
                    'config': self.config.to_dict(),
                    'metrics': self._metrics_to_dict(metrics),
                    'score': score,
                    'action': f"{action.value}:{params}",
                    'improvement': improvement
                }
                report['iterations'].append(iteration_data)
                
                print(f"[Agent] New score: {score:.2%} (improvement: +{improvement:.2%})")
                
                # Update initial for next iteration
                initial_score = score
        
        report['final_config'] = self.config.to_dict()
        report['improvement'] = report['iterations'][-1]['score'] - report['iterations'][0]['score']
        
        return report
    
    def _extract_metrics(self, query_result: Dict) -> MetricsSnapshot:
        """Extract metrics from query result"""
        metrics = query_result.get('metrics', {})
        evaluation = query_result.get('evaluation', {})
        
        generation = evaluation.get('generation', {})
        e2e = evaluation.get('end_to_end', {})
        retrieval = evaluation.get('retrieval', {})
        
        return MetricsSnapshot(
            timestamp=datetime.now(),
            faithfulness=generation.get('faithfulness', 0.5),
            relevance=generation.get('relevance', 0.5),
            hallucination_rate=e2e.get('hallucination_rate', 0.5),
            correctness=e2e.get('correctness', None),
            recall_at_5=retrieval.get('recall@5', None),
            precision_at_5=retrieval.get('precision@5', None),
            mrr=retrieval.get('mrr', None),
            latency_ms=metrics.get('total_time_ms', 0),
            tokens=metrics.get('tokens', 0)
        )
    
    def _metrics_to_dict(self, metrics: MetricsSnapshot) -> Dict:
        """Convert metrics to dict"""
        return {
            'faithfulness': metrics.faithfulness,
            'relevance': metrics.relevance,
            'hallucination_rate': metrics.hallucination_rate,
            'correctness': metrics.correctness,
            'recall_at_5': metrics.recall_at_5,
            'precision_at_5': metrics.precision_at_5,
            'mrr': metrics.mrr,
            'latency_ms': metrics.latency_ms,
            'tokens': metrics.tokens
        }
    
    def _log_action(
        self,
        action: OptimizationAction,
        parameters: Dict,
        success: bool,
        error: Optional[str] = None
    ):
        """Log agent action"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action.value,
            'parameters': parameters,
            'success': success,
            'error': error
        }
        
        self.action_history.append(log_entry)
        
        # Optionally save to file
        try:
            with open(self.history_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"Warning: Could not save action history: {e}")
    
    def get_status(self) -> Dict:
        """Get agent status"""
        if not self.analyzer.metrics_history:
            return {'status': 'no_data'}
        
        latest = self.analyzer.metrics_history[-1]
        
        return {
            'current_score': latest.overall_score(),
            'current_config': self.config.to_dict(),
            'metrics': self._metrics_to_dict(latest),
            'recent_actions': self.action_history[-5:],
            'metrics_trend': {
                'faithfulness': self.analyzer.get_trend('faithfulness'),
                'relevance': self.analyzer.get_trend('relevance'),
                'hallucination': self.analyzer.get_trend('hallucination_rate'),
            }
        }