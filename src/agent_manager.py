"""
Agent Manager - Manages AI Agent lifecycle and operations
"""

import json
import time
from typing import Dict, Optional, List
from datetime import datetime
from threading import Thread
import logging

from src.ai_agent import RAGOptimizationAgent, MetricsSnapshot


logger = logging.getLogger(__name__)


class AgentManager:
    """
    Manages RAG optimization agent
    Handles continuous monitoring and optimization
    """
    
    def __init__(
        self,
        rag_engine,
        llm_client=None,
        config_file: str = "agent_config.json",
        history_file: str = "agent_history.jsonl",
        report_file: str = "agent_reports.jsonl"
    ):
        """
        Initialize agent manager
        
        Args:
            rag_engine: RAGEngine to optimize
            llm_client: Optional LLM for reasoning
            config_file: Path to save agent config
            history_file: Path to save action history
            report_file: Path to save optimization reports
        """
        self.rag_engine = rag_engine
        self.llm_client = llm_client
        
        self.agent = RAGOptimizationAgent(
            rag_engine,
            llm_client,
            history_file=history_file
        )
        
        self.config_file = config_file
        self.report_file = report_file
        
        self.is_running = False
        self.monitoring_thread: Optional[Thread] = None
        
        self.stats = {
            'total_optimizations': 0,
            'total_actions': 0,
            'successful_improvements': 0,
            'total_score_improvement': 0.0,
        }
        
        self._load_stats()
    
    def start_continuous_monitoring(
        self,
        test_queries: List[str],
        interval_seconds: int = 300,
        max_iterations_per_cycle: int = 3
    ):
        """
        Start continuous monitoring and optimization
        
        Args:
            test_queries: Queries to use for evaluation
            interval_seconds: Seconds between optimization cycles
            max_iterations_per_cycle: Max iterations per cycle
        """
        self.is_running = True
        
        self.monitoring_thread = Thread(
            target=self._monitoring_loop,
            args=(test_queries, interval_seconds, max_iterations_per_cycle),
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info(f"Agent monitoring started (interval: {interval_seconds}s)")
    
    def stop_continuous_monitoring(self):
        """Stop continuous monitoring"""
        self.is_running = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        
        logger.info("Agent monitoring stopped")
    
    def _monitoring_loop(
        self,
        test_queries: List[str],
        interval_seconds: int,
        max_iterations: int
    ):
        """
        Main monitoring loop (runs in background thread)
        
        Args:
            test_queries: List of test queries
            interval_seconds: Sleep interval between cycles
            max_iterations: Max optimization iterations per cycle
        """
        cycle = 0
        
        while self.is_running:
            cycle += 1
            
            try:
                logger.info(f"[Agent] Starting optimization cycle {cycle}")
                
                # Select test query (round-robin)
                test_query = test_queries[cycle % len(test_queries)]
                
                # Run optimization
                report = self.agent.optimize(
                    test_query=test_query,
                    max_iterations=max_iterations
                )
                
                # Process report
                self._process_optimization_report(report)
                
                # Save report
                self._save_report(report)
                
                logger.info(f"[Agent] Cycle {cycle} completed. "
                           f"Improvement: {report['improvement']:.2%}")
                
            except Exception as e:
                logger.error(f"Error in monitoring cycle {cycle}: {str(e)}")
            
            # Wait before next cycle
            if self.is_running:
                time.sleep(interval_seconds)
    
    def _process_optimization_report(self, report: Dict):
        """Process optimization report and update stats"""
        self.stats['total_optimizations'] += 1
        
        improvement = report.get('improvement', 0)
        self.stats['total_score_improvement'] += improvement
        
        for iteration in report.get('iterations', [])[1:]:  # Skip initial
            action = iteration.get('action')
            if action:
                self.stats['total_actions'] += 1
                if iteration.get('improvement', 0) > 0:
                    self.stats['successful_improvements'] += 1
        
        self._save_stats()
    
    def _save_report(self, report: Dict):
        """Save optimization report"""
        try:
            with open(self.report_file, 'a') as f:
                f.write(json.dumps(report) + '\n')
        except Exception as e:
            logger.error(f"Error saving report: {e}")
    
    def _save_stats(self):
        """Save statistics"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump({
                    'stats': self.stats,
                    'timestamp': datetime.now().isoformat(),
                    'agent_config': self.agent.config.to_dict()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving stats: {e}")
    
    def _load_stats(self):
        """Load previous statistics"""
        try:
            with open(self.config_file, 'r') as f:
                data = json.load(f)
                self.stats = data.get('stats', self.stats)
                logger.info(f"Loaded agent stats: {self.stats}")
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.error(f"Error loading stats: {e}")
    
    def run_single_optimization(
        self,
        test_query: str,
        max_iterations: int = 3
    ) -> Dict:
        """
        Run single optimization cycle
        
        Args:
            test_query: Query to optimize on
            max_iterations: Max iterations
            
        Returns: Optimization report
        """
        try:
            report = self.agent.optimize(
                test_query=test_query,
                max_iterations=max_iterations
            )
            
            self._process_optimization_report(report)
            self._save_report(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error running optimization: {e}")
            return {'error': str(e)}
    
    def get_agent_status(self) -> Dict:
        """Get current agent status"""
        return {
            'is_running': self.is_running,
            'stats': self.stats,
            'agent_status': self.agent.get_status(),
            'timestamp': datetime.now().isoformat()
        }
    
    def recommend_actions(self) -> List[Dict]:
        """Get recommended optimization actions"""
        actions = self.agent.analyze_and_plan()
        
        return [
            {
                'action': action[0].value,
                'parameters': action[1],
                'description': self._get_action_description(action[0], action[1])
            }
            for action in actions
        ]
    
    @staticmethod
    def _get_action_description(action, params: Dict) -> str:
        """Get human-readable description of action"""
        descriptions = {
            'adjust_top_k': f"Increase context retrieval from {params.get('top_k', 'N/A')} documents",
            'adjust_alpha': f"Adjust semantic weight to {params.get('alpha', 0.7):.1f}",
            'change_retrieval_method': f"Switch to {params.get('method', 'hybrid')} retrieval",
            'adjust_temperature': f"Reduce creativity to {params.get('temperature', 0.5)}",
            'adjust_max_tokens': f"Limit response length to {params.get('max_tokens', 1000)} tokens",
            'increase_context': f"Increase top-K to {params.get('top_k', 'N/A')}",
            'add_system_prompt': f"Emphasize {params.get('emphasis', 'default')} in instructions",
            'enable_hybrid': "Enable hybrid search for better accuracy",
            'no_action': "System is performing optimally"
        }
        
        action_name = action.value if hasattr(action, 'value') else str(action)
        return descriptions.get(action_name, f"Execute {action_name}")
    
    def apply_recommendation(self, action_name: str, params: Dict) -> bool:
        """
        Apply a recommended action
        
        Args:
            action_name: Name of action to apply
            params: Parameters for action
            
        Returns: Success flag
        """
        from src.ai_agent import OptimizationAction
        
        try:
            action = OptimizationAction[action_name.upper()]
            success = self.agent.execute_action(action, params)
            
            if success:
                logger.info(f"Applied action: {action_name}")
            
            return success
            
        except KeyError:
            logger.error(f"Unknown action: {action_name}")
            return False
    
    def get_optimization_history(self, limit: int = 10) -> List[Dict]:
        """Get recent optimization reports"""
        reports = []
        
        try:
            with open(self.report_file, 'r') as f:
                lines = f.readlines()
                for line in lines[-limit:]:
                    reports.append(json.loads(line))
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.error(f"Error reading history: {e}")
        
        return reports
    
    def export_report(self, filepath: str) -> bool:
        """Export comprehensive agent report"""
        try:
            report = {
                'generated_at': datetime.now().isoformat(),
                'status': self.get_agent_status(),
                'recommendations': self.recommend_actions(),
                'history': self.get_optimization_history(limit=50),
                'current_config': self.agent.config.to_dict()
            }
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Report exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
            return False