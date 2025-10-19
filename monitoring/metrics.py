from prometheus_client import (
    Counter, Histogram, Gauge, Info, Summary, CollectorRegistry
)
import time
from functools import wraps

# === Safe global registry (avoid duplicate registration errors) ===
custom_registry = CollectorRegistry(auto_describe=True)

# === Core Metrics ===
query_counter = Counter(
    'rag_queries_total',
    'Total number of RAG queries',
    ['retrieval_method', 'status'],
    registry=custom_registry
)

retrieval_latency = Histogram(
    'rag_retrieval_duration_seconds',
    'Time spent on document retrieval',
    ['retrieval_method'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    registry=custom_registry
)

generation_latency = Histogram(
    'rag_generation_duration_seconds',
    'Time spent on LLM generation',
    ['model'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    registry=custom_registry
)

total_latency = Histogram(
    'rag_total_duration_seconds',
    'Total RAG query time',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    registry=custom_registry
)

tokens_used = Counter(
    'rag_tokens_total',
    'Total tokens consumed',
    ['model', 'type'],  # type = prompt/completion
    registry=custom_registry
)

documents_retrieved = Histogram(
    'rag_documents_retrieved',
    'Number of documents retrieved per query',
    buckets=[1, 2, 3, 5, 10],
    registry=custom_registry
)

indexed_documents = Gauge(
    'rag_indexed_documents',
    'Number of documents in vector store',
    registry=custom_registry
)

active_queries = Gauge(
    'rag_active_queries',
    'Number of queries currently being processed',
    registry=custom_registry
)

system_info = Info(
    'rag_system',
    'RAG system information',
    registry=custom_registry
)

# === Evaluation Metrics ===
recall_at_k = Summary(
    'rag_retrieval_recall_at_k',
    'Recall@K for retrieval',
    ['k'],
    registry=custom_registry
)

precision_at_k = Summary(
    'rag_retrieval_precision_at_k',
    'Precision@K for retrieval',
    ['k'],
    registry=custom_registry
)

mrr_metric = Summary(
    'rag_retrieval_mrr',
    'Mean Reciprocal Rank',
    registry=custom_registry
)

ndcg_at_k = Summary(
    'rag_retrieval_ndcg_at_k',
    'nDCG@K for retrieval',
    ['k'],
    registry=custom_registry
)

average_precision_metric = Summary(
    'rag_retrieval_average_precision',
    'Average Precision for retrieval',
    registry=custom_registry
)

faithfulness_metric = Summary(
    'rag_generation_faithfulness',
    'Faithfulness of generated answers',
    registry=custom_registry
)

relevance_metric = Summary(
    'rag_generation_relevance',
    'Relevance of answers to questions',
    registry=custom_registry
)

correctness_metric = Summary(
    'rag_e2e_correctness',
    'Correctness of answers vs ground truth',
    registry=custom_registry
)

hallucination_rate_metric = Summary(
    'rag_e2e_hallucination_rate',
    'Rate of hallucinated content',
    registry=custom_registry
)

overall_quality_score = Summary(
    'rag_e2e_overall_quality',
    'Overall quality score (composite)',
    registry=custom_registry
)

# === Metrics Collector Class ===
class MetricsCollector:
    """Helper class to collect and record metrics"""

    # --- Core ---
    @staticmethod
    def record_query(retrieval_method: str, status: str = "success"):
        query_counter.labels(retrieval_method=retrieval_method, status=status).inc()

    @staticmethod
    def record_retrieval_time(duration: float, method: str):
        retrieval_latency.labels(retrieval_method=method).observe(duration)

    @staticmethod
    def record_generation_time(duration: float, model: str):
        generation_latency.labels(model=model).observe(duration)

    @staticmethod
    def record_total_time(duration: float):
        total_latency.observe(duration)

    @staticmethod
    def record_tokens(count: int, model: str, token_type: str = "completion"):
        tokens_used.labels(model=model, type=token_type).inc(count)

    @staticmethod
    def record_documents_retrieved(count: int):
        documents_retrieved.observe(count)

    @staticmethod
    def set_indexed_documents(count: int):
        indexed_documents.set(count)

    @staticmethod
    def set_system_info(info: dict):
        system_info.info(info)

    # --- Evaluation: Retriever ---
    @staticmethod
    def record_retrieval_metrics(metrics: dict):
        for k in ['3', '5', '10']:
            if f"recall@{k}" in metrics:
                recall_at_k.labels(k=k).observe(metrics[f"recall@{k}"])
            if f"precision@{k}" in metrics:
                precision_at_k.labels(k=k).observe(metrics[f"precision@{k}"])
            if f"ndcg@{k}" in metrics:
                ndcg_at_k.labels(k=k).observe(metrics[f"ndcg@{k}"])
        if 'mrr' in metrics:
            mrr_metric.observe(metrics['mrr'])
        if 'average_precision' in metrics:
            average_precision_metric.observe(metrics['average_precision'])

    # --- Evaluation: Generator ---
    @staticmethod
    def record_generation_metrics(metrics: dict):
        if 'faithfulness' in metrics:
            faithfulness_metric.observe(metrics['faithfulness'])
        if 'relevance' in metrics:
            relevance_metric.observe(metrics['relevance'])

    # --- Evaluation: End-to-End ---
    @staticmethod
    def record_e2e_metrics(metrics: dict):
        if 'correctness' in metrics:
            correctness_metric.observe(metrics['correctness'])
        if 'hallucination_rate' in metrics:
            hallucination_rate_metric.observe(metrics['hallucination_rate'])

    @staticmethod
    def record_overall_quality(score: float):
        overall_quality_score.observe(score)


# === Decorator for Monitoring Queries ===
def monitor_query(func):
    """Decorator to wrap RAG query functions for automatic metric recording"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        active_queries.inc()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time

            metrics = result.get('metrics', {})
            retrieval_method = metrics.get('retrieval_method', 'unknown')
            model = metrics.get('model', 'unknown')

            # --- Record core metrics ---
            MetricsCollector.record_query(retrieval_method, "success")
            MetricsCollector.record_total_time(duration)
            if 'retrieval_time_ms' in metrics:
                MetricsCollector.record_retrieval_time(metrics['retrieval_time_ms'] / 1000, retrieval_method)
            if 'generation_time_ms' in metrics:
                MetricsCollector.record_generation_time(metrics['generation_time_ms'] / 1000, model)
            if 'tokens' in metrics:
                MetricsCollector.record_tokens(metrics['tokens'], model)
            if 'num_retrieved' in metrics:
                MetricsCollector.record_documents_retrieved(metrics['num_retrieved'])

            # --- Record evaluation metrics if available ---
            eval_metrics = result.get('evaluation', {})
            if 'retrieval' in eval_metrics:
                MetricsCollector.record_retrieval_metrics(eval_metrics['retrieval'])
            if 'generation' in eval_metrics:
                MetricsCollector.record_generation_metrics(eval_metrics['generation'])
            if 'end_to_end' in eval_metrics:
                MetricsCollector.record_e2e_metrics(eval_metrics['end_to_end'])
            if 'overall_score' in eval_metrics:
                MetricsCollector.record_overall_quality(eval_metrics['overall_score'])

            return result

        except Exception:
            MetricsCollector.record_query('unknown', 'error')
            raise
        finally:
            active_queries.dec()
    return wrapper
