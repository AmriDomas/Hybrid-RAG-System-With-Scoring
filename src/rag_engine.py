from typing import Dict, List, Optional, Set
from src.vector_store import VectorStore
from src.llm_client import LLMClient
from src.evaluation_metrics import RAGEvaluator
import time

class RAGEngine:
    """Main RAG engine orchestrating retrieval and generation"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        llm_client: LLMClient,
        top_k: int = 3,
        hybrid_alpha: float = 0.7,
        enable_evaluation: bool = False,
        use_llm_judge: bool = False
    ):
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.top_k = top_k
        self.hybrid_alpha = hybrid_alpha
        self.enable_evaluation = enable_evaluation
        
        # Initialize evaluator
        if enable_evaluation:
            self.evaluator = RAGEvaluator(
                llm_client=llm_client if use_llm_judge else None,
                use_llm_judge=use_llm_judge
            )
        else:
            self.evaluator = None
        
    def query(
        self,
        question: str,
        retrieval_method: str = "hybrid",
        ground_truth: Optional[str] = None,
        relevant_doc_ids: Optional[Set[str]] = None,
        relevance_scores: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        End-to-end RAG query with optional evaluation
        
        Args:
            question: User query
            retrieval_method: 'semantic', 'keyword', or 'hybrid'
            ground_truth: Optional reference answer for evaluation
            relevant_doc_ids: Optional set of relevant doc IDs for retrieval eval
            relevance_scores: Optional dict of doc_id -> relevance for nDCG
            
        Returns:
            Dict with answer, retrieved docs, metrics, and optional evaluation
        """
        start_time = time.time()
        
        # Retrieval phase
        if retrieval_method == "semantic":
            retrieved_docs = self.vector_store.semantic_search(
                question, top_k=self.top_k
            )
        elif retrieval_method == "keyword":
            retrieved_docs = self.vector_store.keyword_search(
                question, top_k=self.top_k
            )
        else:  # hybrid
            retrieved_docs = self.vector_store.hybrid_search(
                question, top_k=self.top_k, alpha=self.hybrid_alpha
            )
        
        retrieval_time = time.time() - start_time
        
        # Generation phase
        generation_result = self.llm_client.generate(
            prompt=question,
            context=retrieved_docs
        )
        
        total_time = time.time() - start_time
        
        result = {
            "question": question,
            "answer": generation_result["response"],
            "retrieved_documents": retrieved_docs,
            "metrics": {
                "retrieval_method": retrieval_method,
                "num_retrieved": len(retrieved_docs),
                "retrieval_time_ms": retrieval_time * 1000,
                "generation_time_ms": generation_result["latency"] * 1000,
                "total_time_ms": total_time * 1000,
                "tokens": generation_result.get("tokens", 0),
                "prompt_tokens": generation_result.get("prompt_tokens", 0),
                "completion_tokens": generation_result.get("completion_tokens", 0),
                "model": generation_result["model"]
            }
        }
        
        # Evaluation phase (if enabled)
        if self.enable_evaluation and self.evaluator:
            retrieved_doc_ids = [doc['id'] for doc in retrieved_docs]
            
            evaluation = self.evaluator.evaluate_full_pipeline(
                question=question,
                answer=generation_result["response"],
                retrieved_doc_ids=retrieved_doc_ids,
                context_docs=retrieved_docs,
                relevant_doc_ids=relevant_doc_ids,
                ground_truth=ground_truth,
                relevance_scores=relevance_scores
            )
            
            result['evaluation'] = evaluation
        
        return result
    
    def query_with_evaluation(
        self,
        question: str,
        retrieval_method: str = "hybrid",
        ground_truth: Optional[str] = None,
        relevant_doc_ids: Optional[Set[str]] = None
    ) -> Dict:
        """
        Convenience method for query with evaluation enabled
        """
        old_eval_state = self.enable_evaluation
        self.enable_evaluation = True
        
        if not self.evaluator:
            self.evaluator = RAGEvaluator(llm_client=self.llm_client)
        
        result = self.query(
            question=question,
            retrieval_method=retrieval_method,
            ground_truth=ground_truth,
            relevant_doc_ids=relevant_doc_ids
        )
        
        self.enable_evaluation = old_eval_state
        return result
    
    def batch_query(self, questions: List[str]) -> List[Dict]:
        """Process multiple queries"""
        results = []
        for q in questions:
            results.append(self.query(q))
        return results
    
    def batch_evaluate(
        self,
        test_cases: List[Dict]
    ) -> Dict:
        """
        Batch evaluation on test set
        
        Args:
            test_cases: List of dicts with 'question', 'ground_truth', 'relevant_docs'
            
        Returns:
            Aggregated metrics
        """
        if not self.evaluator:
            self.evaluator = RAGEvaluator(llm_client=self.llm_client)
        
        all_results = []
        
        for test_case in test_cases:
            result = self.query_with_evaluation(
                question=test_case['question'],
                ground_truth=test_case.get('ground_truth'),
                relevant_doc_ids=set(test_case.get('relevant_docs', []))
            )
            all_results.append(result)
        
        # Aggregate metrics
        aggregated = self._aggregate_evaluation_metrics(all_results)
        
        return {
            'num_test_cases': len(test_cases),
            'individual_results': all_results,
            'aggregated_metrics': aggregated
        }
    
    def _aggregate_evaluation_metrics(self, results: List[Dict]) -> Dict:
        """Aggregate evaluation metrics across multiple queries"""
        aggregated = {
            'retrieval': {},
            'generation': {},
            'end_to_end': {},
            'performance': {}
        }
        
        # Collect all metrics
        retrieval_metrics = []
        generation_metrics = []
        e2e_metrics = []
        
        for result in results:
            if 'evaluation' not in result:
                continue
            
            eval_data = result['evaluation']
            
            if 'retrieval' in eval_data:
                retrieval_metrics.append(eval_data['retrieval'])
            if 'generation' in eval_data:
                generation_metrics.append(eval_data['generation'])
            if 'end_to_end' in eval_data:
                e2e_metrics.append(eval_data['end_to_end'])
        
        # Average retrieval metrics
        if retrieval_metrics:
            for metric in ['recall@3', 'recall@5', 'recall@10', 'precision@3', 
                          'precision@5', 'precision@10', 'mrr', 'ndcg@3', 
                          'ndcg@5', 'ndcg@10', 'average_precision']:
                values = [m.get(metric, 0) for m in retrieval_metrics if metric in m]
                if values:
                    aggregated['retrieval'][metric] = sum(values) / len(values)
        
        # Average generation metrics
        if generation_metrics:
            for metric in ['faithfulness', 'relevance']:
                values = [m.get(metric, 0) for m in generation_metrics if metric in m]
                if values:
                    aggregated['generation'][metric] = sum(values) / len(values)
        
        # Average end-to-end metrics
        if e2e_metrics:
            for metric in ['correctness', 'hallucination_rate']:
                values = [m.get(metric, 0) for m in e2e_metrics if metric in m]
                if values:
                    aggregated['end_to_end'][metric] = sum(values) / len(values)
        
        # Performance metrics
        latencies = [r['metrics']['total_time_ms'] for r in results]
        tokens = [r['metrics']['tokens'] for r in results]
        
        aggregated['performance'] = {
            'avg_latency_ms': sum(latencies) / len(latencies),
            'p50_latency_ms': sorted(latencies)[len(latencies)//2],
            'p95_latency_ms': sorted(latencies)[int(len(latencies)*0.95)],
            'avg_tokens': sum(tokens) / len(tokens),
            'total_tokens': sum(tokens)
        }
        
        # Overall quality score
        quality_scores = []
        if 'faithfulness' in aggregated['generation']:
            quality_scores.append(aggregated['generation']['faithfulness'])
        if 'relevance' in aggregated['generation']:
            quality_scores.append(aggregated['generation']['relevance'])
        if 'hallucination_rate' in aggregated['end_to_end']:
            quality_scores.append(1 - aggregated['end_to_end']['hallucination_rate'])
        if 'correctness' in aggregated['end_to_end']:
            quality_scores.append(aggregated['end_to_end']['correctness'])
        
        if quality_scores:
            aggregated['overall_quality_score'] = sum(quality_scores) / len(quality_scores)
        
        return aggregated
    
    def get_system_status(self) -> Dict:
        """Get system health metrics"""
        return {
            "status": "healthy",
            "documents_indexed": len(self.vector_store.documents),
            "embedding_model": self.vector_store.encoder.get_sentence_embedding_dimension(),
            "llm_model": self.llm_client.model,
            "evaluation_enabled": self.enable_evaluation
        }