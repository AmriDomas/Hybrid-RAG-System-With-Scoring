"""
Complete RAG metrics for Prometheus - CONSISTENT CompleteMetricsManager
"""

import logging
from typing import Dict, Optional
from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest, CollectorRegistry, Enum

logger = logging.getLogger(__name__)

# Create separate registry for RAG metrics
rag_registry = CollectorRegistry()

class CompleteMetricsManager:
    """
    Complete metrics manager for RAG system - CONSISTENT VERSION
    """
    
    def __init__(self):
        logger.info("Initializing CompleteMetricsManager")
        self._init_prometheus_metrics()
        logger.info("All Prometheus metrics registered")
    
    def _init_prometheus_metrics(self):
        """Initialize all Prometheus metric collectors"""
        
        # === 1. QUERY & VOLUME METRICS ===
        self.query_counter = Counter(
            'rag_queries_total',
            'Total number of RAG queries',
            ['retrieval_method', 'status', 'has_evaluation'],
            registry=rag_registry
        )
        
        self.active_queries = Gauge(
            'rag_active_queries',
            'Number of queries currently being processed',
            registry=rag_registry
        )
        
        # === 2. LATENCY & PERFORMANCE METRICS ===
        self.retrieval_latency = Histogram(
            'rag_retrieval_duration_seconds',
            'Time spent on document retrieval',
            ['retrieval_method'],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=rag_registry
        )
        
        self.generation_latency = Histogram(
            'rag_generation_duration_seconds',
            'Time spent on LLM generation',
            ['model'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0],
            registry=rag_registry
        )
        
        self.total_latency = Histogram(
            'rag_total_duration_seconds',
            'Total RAG query time',
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0],
            registry=rag_registry
        )
        
        # === 3. RETRIEVAL QUALITY METRICS ===
        self.recall_at_k = Summary(
            'rag_retrieval_recall_at_k',
            'Recall@K for retrieval',
            ['k'],
            registry=rag_registry
        )
        
        self.precision_at_k = Summary(
            'rag_retrieval_precision_at_k',
            'Precision@K for retrieval',
            ['k'],
            registry=rag_registry
        )
        
        self.mrr_metric = Summary(
            'rag_retrieval_mrr',
            'Mean Reciprocal Rank for retrieval',
            registry=rag_registry
        )
        
        self.ndcg_at_k = Summary(
            'rag_retrieval_ndcg_at_k',
            'nDCG@K for retrieval',
            ['k'],
            registry=rag_registry
        )
        
        # === 4. GENERATION QUALITY METRICS ===
        self.faithfulness = Summary(
            'rag_generation_faithfulness',
            'Faithfulness of generated answers (0-1)',
            registry=rag_registry
        )
        
        self.relevance = Summary(
            'rag_generation_relevance',
            'Relevance of answers to questions (0-1)',
            registry=rag_registry
        )
        
        self.answer_length = Histogram(
            'rag_generation_answer_length',
            'Length of generated answers in characters',
            buckets=[50, 100, 200, 500, 1000, 2000, 5000],
            registry=rag_registry
        )
        
        # === 5. END-TO-END QUALITY METRICS ===
        self.correctness = Summary(
            'rag_e2e_correctness',
            'Correctness of answers vs ground truth (0-1)',
            registry=rag_registry
        )
        
        self.hallucination_rate = Summary(
            'rag_e2e_hallucination_rate',
            'Rate of hallucinated content (0-1)',
            registry=rag_registry
        )
        
        self.overall_quality = Summary(
            'rag_e2e_overall_quality',
            'Overall quality score (composite 0-1)',
            registry=rag_registry
        )
        
        # === 6. RESOURCE & COST METRICS ===
        self.tokens_used = Counter(
            'rag_tokens_total',
            'Total tokens consumed',
            ['model', 'type'],
            registry=rag_registry
        )
        
        self.documents_retrieved = Histogram(
            'rag_documents_retrieved',
            'Number of documents retrieved per query',
            buckets=[1, 2, 3, 5, 7, 10, 15, 20],
            registry=rag_registry
        )
        
        self.indexed_documents = Gauge(
            'rag_indexed_documents',
            'Number of documents in vector store',
            registry=rag_registry
        )
        
        # === 7. SYSTEM & HEALTH METRICS ===
        self.system_health = Enum(
            'rag_system_health',
            'Overall system health status',
            states=['healthy', 'degraded', 'unhealthy'],
            registry=rag_registry
        )
        
        self.embedding_model_info = Gauge(
            'rag_embedding_model_info',
            'Embedding model information',
            ['model_name', 'dimension'],
            registry=rag_registry
        )
        
        self.llm_model_info = Gauge(
            'rag_llm_model_info',
            'LLM model information',
            ['model_name', 'version'],
            registry=rag_registry
        )
        
        # Set initial health
        self.system_health.state('healthy')
    
    def record_query_start(self):
        """Record query start for concurrent tracking"""
        self.active_queries.inc()
    
    def record_query_end(self):
        """Record query end for concurrent tracking"""
        self.active_queries.dec()
    
    def record_query_metrics(
        self,
        retrieval_method: str,
        latency_ms: float,
        tokens: int,
        num_documents: int,  # PARAMETER YANG DITAMBAHKAN
        answer_length: int,
        metrics: Dict,
        status: str = "success",
        has_evaluation: bool = False
    ) -> None:
        """
        Record comprehensive metrics for a query - FIXED VERSION WITH num_documents
        """
        try:
            logger.info(f"Recording metrics for {retrieval_method} query...")
            
            # 1. Query volume
            self.query_counter.labels(
                retrieval_method=retrieval_method or "unknown",
                status=status or "unknown",
                has_evaluation=str(has_evaluation).lower()
            ).inc()
            
            # 2. Latency metrics
            latency_seconds = latency_ms / 1000.0
            self.total_latency.observe(latency_seconds)
            
            # 3. Resource metrics - GUNAKAN num_documents
            self.tokens_used.labels(model='gpt-4', type='total').inc(tokens)
            self.documents_retrieved.observe(num_documents)  # GUNAKAN PARAMETER BARU
            self.answer_length.observe(answer_length)
            
            # 4. Quality metrics (if available)
            if metrics:
                if 'faithfulness' in metrics and metrics['faithfulness'] is not None:
                    self.faithfulness.observe(float(metrics['faithfulness']))
                
                if 'relevance' in metrics and metrics['relevance'] is not None:
                    self.relevance.observe(float(metrics['relevance']))
                
                if 'hallucination_rate' in metrics and metrics['hallucination_rate'] is not None:
                    self.hallucination_rate.observe(float(metrics['hallucination_rate']))
                
                if 'correctness' in metrics and metrics['correctness'] is not None:
                    self.correctness.observe(float(metrics['correctness']))
                
                if 'overall_score' in metrics and metrics['overall_score'] is not None:
                    self.overall_quality.observe(float(metrics['overall_score']))
                
                # 5. Retrieval metrics (if available)
                retrieval_metrics = metrics.get('retrieval', {})
                if retrieval_metrics:
                    for k in ['3', '5', '10']:
                        recall_key = f'recall@{k}'
                        if recall_key in retrieval_metrics and retrieval_metrics[recall_key] is not None:
                            self.recall_at_k.labels(k=k).observe(float(retrieval_metrics[recall_key]))
                        
                        precision_key = f'precision@{k}'
                        if precision_key in retrieval_metrics and retrieval_metrics[precision_key] is not None:
                            self.precision_at_k.labels(k=k).observe(float(retrieval_metrics[precision_key]))
                        
                        ndcg_key = f'ndcg@{k}'
                        if ndcg_key in retrieval_metrics and retrieval_metrics[ndcg_key] is not None:
                            self.ndcg_at_k.labels(k=k).observe(float(retrieval_metrics[ndcg_key]))
                    
                    if 'mrr' in retrieval_metrics and retrieval_metrics['mrr'] is not None:
                        self.mrr_metric.observe(float(retrieval_metrics['mrr']))
            
            logger.info(f"✅ Successfully recorded metrics for {retrieval_method} query")
            
        except Exception as e:
            logger.error(f"❌ Error recording complete metrics: {e}")
            logger.error(f"❌ Error details: retrieval_method={retrieval_method}, metrics_keys={list(metrics.keys()) if metrics else 'None'}")
            raise
    
    def set_system_info(
        self,
        version: str,
        embedding_model: str,
        embedding_dimension: int,
        llm_model: str,
        document_count: int
    ):
        """Set system information"""
        try:
            self.indexed_documents.set(document_count)
            self.embedding_model_info.labels(
                model_name=embedding_model,
                dimension=str(embedding_dimension)
            ).set(1)
            self.llm_model_info.labels(
                model_name=llm_model,
                version=version
            ).set(1)
            logger.info(f"✅ Set system info: {document_count} docs, {embedding_model}, {llm_model}")
        except Exception as e:
            logger.error(f"❌ Error setting system info: {e}")
    
    def set_system_health(self, status: str):
        """Set system health status"""
        try:
            self.system_health.state(status)
            logger.info(f"✅ System health set to: {status}")
        except Exception as e:
            logger.error(f"❌ Error setting system health: {e}")
    
    def get_all_metrics_prometheus_format(self) -> bytes:
        """Get all RAG metrics in Prometheus format"""
        try:
            metrics_data = generate_latest(rag_registry)
            logger.info(f"Generated {len(metrics_data)} bytes of metrics data")
            return metrics_data
        except Exception as e:
            logger.error(f"❌ Error generating metrics: {e}")
            return b"# Error generating metrics\n"

# Global instance - GUNAKAN CompleteMetricsManager
_metrics_manager: Optional[CompleteMetricsManager] = None

def get_metrics_manager() -> CompleteMetricsManager:
    """Get or create global metrics manager"""
    global _metrics_manager
    if _metrics_manager is None:
        _metrics_manager = CompleteMetricsManager()
    return _metrics_manager

def record_rag_query(
    retrieval_method: str,
    latency_ms: float,
    tokens: int,
    num_documents: int,
    answer: str,
    metrics: Dict,
    status: str = "success"
) -> None:
    """
    Convenience function to record RAG query metrics - FIXED VERSION
    """
    try:
        manager = get_metrics_manager()
        
        # Start tracking concurrent query
        manager.record_query_start()
        
        # Record the metrics - GUNAKAN PARAMETER YANG KONSISTEN
        manager.record_query_metrics(
            retrieval_method=retrieval_method,
            latency_ms=latency_ms,
            tokens=tokens,
            num_documents=num_documents,
            answer_length=len(answer),
            metrics=metrics,  # Semua metrics termasuk faithfulness, relevance dll harus ada di sini
            status=status,
            has_evaluation=bool(metrics.get('evaluation_enabled', False))
        )
        
        # End tracking concurrent query
        manager.record_query_end()
        
        logger.info(f"✅ Recorded RAG query: {retrieval_method}, {latency_ms}ms, {tokens}tokens")
        
    except Exception as e:
        logger.error(f"❌ Error in record_rag_query: {e}")