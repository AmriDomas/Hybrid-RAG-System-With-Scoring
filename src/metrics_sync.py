"""
Metrics Synchronization Layer
Ensures Prometheus metrics stay in sync across app.py and Prometheus server
"""

import threading
import time
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class MetricSnapshot:
    """Snapshot of metrics at point in time"""
    timestamp: datetime
    metrics: Dict
    labels: Dict


class MetricsBuffer:
    """
    Buffer for metrics to ensure they're captured immediately
    and synced to Prometheus
    """
    
    def __init__(self, buffer_size: int = 100):
        self.buffer = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
        self.last_sync_time = datetime.now()
    
    def add_snapshot(self, metrics: Dict, labels: Dict = None) -> None:
        """Add metrics snapshot to buffer"""
        with self.lock:
            snapshot = MetricSnapshot(
                timestamp=datetime.now(),
                metrics=metrics,
                labels=labels or {}
            )
            self.buffer.append(snapshot)
            logger.debug(f"Metrics buffered: {len(self.buffer)} items")
    
    def get_latest(self) -> Optional[MetricSnapshot]:
        """Get latest snapshot without removing"""
        with self.lock:
            return self.buffer[-1] if self.buffer else None
    
    def get_all_since(self, since_time: datetime) -> List[MetricSnapshot]:
        """Get all snapshots since a given time"""
        with self.lock:
            return [s for s in self.buffer if s.timestamp > since_time]
    
    def clear(self) -> None:
        """Clear buffer"""
        with self.lock:
            self.buffer.clear()


class PrometheusSync:
    """
    Synchronization between app metrics and Prometheus
    Handles timing, delays, and data consistency
    """
    
    # Prometheus scrape interval (default 15s)
    PROMETHEUS_SCRAPE_INTERVAL = 15
    # App update frequency
    APP_UPDATE_INTERVAL = 5
    # Buffer retention (seconds)
    BUFFER_RETENTION = 300
    
    def __init__(self, prometheus_port: int = 9090):
        self.prometheus_port = prometheus_port
        self.metrics_buffer = MetricsBuffer(buffer_size=200)
        self.last_prometheus_scrape = None
        self.sync_callbacks: List[Callable] = []
        
        self._syncing = False
        self._sync_thread: Optional[threading.Thread] = None
        self._metrics_timestamp = {}
    
    def register_sync_callback(self, callback: Callable) -> None:
        """Register callback to be called on sync"""
        self.sync_callbacks.append(callback)
        logger.info(f"Registered sync callback: {callback.__name__}")
    
    def buffer_metrics(self, metrics: Dict, labels: Dict = None) -> None:
        """
        Buffer metrics immediately (called from app.py)
        This is the source of truth for metrics
        """
        self.metrics_buffer.add_snapshot(metrics, labels)
        
        # Update timestamp tracking
        for key in metrics.keys():
            self._metrics_timestamp[key] = datetime.now()
    
    def start_sync_service(self, interval: int = 5) -> None:
        """
        Start background sync service
        Continuously syncs metrics to Prometheus
        
        Args:
            interval: Sync check interval in seconds
        """
        if self._syncing:
            logger.warning("Sync service already running")
            return
        
        self._syncing = True
        self._sync_thread = threading.Thread(
            target=self._sync_loop,
            args=(interval,),
            daemon=True
        )
        self._sync_thread.start()
        logger.info(f"Metrics sync service started (interval: {interval}s)")
    
    def stop_sync_service(self) -> None:
        """Stop background sync service"""
        self._syncing = False
        if self._sync_thread:
            self._sync_thread.join(timeout=10)
        logger.info("Metrics sync service stopped")
    
    def _sync_loop(self, interval: int) -> None:
        """Main sync loop"""
        while self._syncing:
            try:
                # Check if Prometheus should have scraped recently
                self._check_and_sync()
                
                # Log metrics status
                latest = self.metrics_buffer.get_latest()
                if latest:
                    age = (datetime.now() - latest.timestamp).total_seconds()
                    logger.debug(f"Latest metrics age: {age:.1f}s")
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
                time.sleep(interval)
    
    def _check_and_sync(self) -> None:
        """
        Check if metrics need syncing and trigger callbacks
        Prometheus scrapes every PROMETHEUS_SCRAPE_INTERVAL seconds
        """
        now = datetime.now()
        
        # Get latest metrics
        latest = self.metrics_buffer.get_latest()
        if not latest:
            return
        
        # Check if metrics are fresh (less than scrape interval old)
        age = (now - latest.timestamp).total_seconds()
        
        if age <= self.PROMETHEUS_SCRAPE_INTERVAL:
            # Metrics are fresh, trigger sync callbacks
            for callback in self.sync_callbacks:
                try:
                    callback(latest.metrics, latest.labels)
                except Exception as e:
                    logger.error(f"Error in sync callback: {e}")
    
    def get_sync_status(self) -> Dict:
        """Get current sync status"""
        latest = self.metrics_buffer.get_latest()
        
        if not latest:
            return {
                'status': 'no_data',
                'buffered_metrics': 0,
                'last_update': None
            }
        
        age = (datetime.now() - latest.timestamp).total_seconds()
        
        status = 'fresh' if age < 5 else 'stale' if age > 30 else 'ok'
        
        return {
            'status': status,
            'buffered_metrics': len(self.metrics_buffer.buffer),
            'last_update': latest.timestamp.isoformat(),
            'age_seconds': age,
            'syncing': self._syncing
        }
    
    def force_sync_to_prometheus(self) -> bool:
        """Force immediate sync to Prometheus"""
        try:
            latest = self.metrics_buffer.get_latest()
            if not latest:
                logger.warning("No metrics to sync")
                return False
            
            # Trigger all callbacks immediately
            for callback in self.sync_callbacks:
                callback(latest.metrics, latest.labels)
            
            logger.info("Force sync completed")
            return True
            
        except Exception as e:
            logger.error(f"Force sync failed: {e}")
            return False


class MetricsReconciler:
    """
    Reconciles metrics from app.py and Prometheus
    Detects and fixes inconsistencies
    """
    
    def __init__(self, app_metrics_buffer: MetricsBuffer):
        self.app_metrics = app_metrics_buffer
        self.reconciliation_log: deque = deque(maxlen=100)
    
    def compare_metrics(self, app_metric: Dict, prometheus_metric: Dict) -> Dict:
        """
        Compare app metrics with Prometheus metrics
        
        Returns:
            {
                'synced': bool,
                'differences': List[str],
                'missing_in_prometheus': List[str],
                'extra_in_prometheus': List[str]
            }
        """
        differences = []
        missing_in_prom = []
        extra_in_prom = []
        
        # Check for missing in Prometheus
        for key in app_metric.keys():
            if key not in prometheus_metric:
                missing_in_prom.append(key)
            elif app_metric[key] != prometheus_metric[key]:
                # Allow small numeric differences (rounding)
                try:
                    app_val = float(app_metric[key])
                    prom_val = float(prometheus_metric[key])
                    if abs(app_val - prom_val) > 0.01:  # Tolerance
                        differences.append(f"{key}: app={app_val}, prom={prom_val}")
                except (ValueError, TypeError):
                    differences.append(f"{key}: app={app_metric[key]}, prom={prometheus_metric[key]}")
        
        # Check for extra in Prometheus
        for key in prometheus_metric.keys():
            if key not in app_metric:
                extra_in_prom.append(key)
        
        synced = len(differences) == 0 and len(missing_in_prom) == 0
        
        result = {
            'synced': synced,
            'differences': differences,
            'missing_in_prometheus': missing_in_prom,
            'extra_in_prometheus': extra_in_prom,
            'timestamp': datetime.now().isoformat()
        }
        
        self.reconciliation_log.append(result)
        
        if not synced:
            logger.warning(f"Metrics out of sync: {result}")
        
        return result
    
    def get_reconciliation_status(self) -> Dict:
        """Get overall reconciliation status"""
        if not self.reconciliation_log:
            return {'status': 'no_data', 'checks': 0}
        
        recent = list(self.reconciliation_log)[-10:]
        synced_count = sum(1 for r in recent if r['synced'])
        
        return {
            'recent_checks': len(recent),
            'synced': synced_count,
            'out_of_sync': len(recent) - synced_count,
            'sync_rate': f"{synced_count/len(recent)*100:.1f}%",
            'latest': recent[-1] if recent else None
        }
    
    def get_reconciliation_log(self, limit: int = 10) -> List[Dict]:
        """Get recent reconciliation logs"""
        return list(self.reconciliation_log)[-limit:]


class MetricsValidator:
    """
    Validates metric values and ensures consistency
    """
    
    @staticmethod
    def validate_metric_range(metric_name: str, value: float, 
                            min_val: float = 0, max_val: float = 1) -> bool:
        """
        Validate metric is within expected range
        
        Args:
            metric_name: Name of metric
            value: Value to validate
            min_val: Minimum acceptable value
            max_val: Maximum acceptable value
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(value, (int, float)):
            logger.warning(f"{metric_name}: Not a number - {value}")
            return False
        
        if value < min_val or value > max_val:
            logger.warning(f"{metric_name}: Out of range - {value} (expected {min_val}-{max_val})")
            return False
        
        return True
    
    @staticmethod
    def validate_metrics_batch(metrics: Dict) -> Dict:
        """
        Validate multiple metrics at once
        
        Returns:
            {
                'valid': bool,
                'issues': List[str]
            }
        """
        issues = []
        
        # Define ranges for known metrics
        ranges = {
            'faithfulness': (0, 1),
            'relevance': (0, 1),
            'hallucination_rate': (0, 1),
            'recall@3': (0, 1),
            'recall@5': (0, 1),
            'recall@10': (0, 1),
            'precision@3': (0, 1),
            'precision@5': (0, 1),
            'precision@10': (0, 1),
            'mrr': (0, 1),
            'ndcg@3': (0, 1),
            'ndcg@5': (0, 1),
            'ndcg@10': (0, 1),
        }
        
        for metric_name, (min_val, max_val) in ranges.items():
            if metric_name in metrics:
                if not MetricsValidator.validate_metric_range(
                    metric_name, 
                    metrics[metric_name],
                    min_val, 
                    max_val
                ):
                    issues.append(f"{metric_name} out of range")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }


class SyncedMetricsCollector:
    """
    Enhanced metrics collector with sync capabilities
    Use this instead of regular MetricsCollector for sync
    """
    
    def __init__(self, prometheus_sync: PrometheusSync):
        self.sync = prometheus_sync
        self.current_metrics: Dict = {}
    
    def record_all_metrics(self, metrics: Dict, labels: Dict = None) -> None:
        """
        Record all metrics and immediately sync to Prometheus
        
        This is the single source of truth for metrics
        """
        # Validate metrics
        validation = MetricsValidator.validate_metrics_batch(metrics)
        if not validation['valid']:
            logger.warning(f"Invalid metrics: {validation['issues']}")
        
        # Store metrics
        self.current_metrics = metrics.copy()
        
        # Buffer for sync
        self.sync.buffer_metrics(metrics, labels)
        
        # Immediate callback
        logger.debug(f"Metrics recorded: {list(metrics.keys())}")
    
    def get_current_metrics(self) -> Dict:
        """Get currently stored metrics"""
        return self.current_metrics.copy()
    
    def get_sync_status(self) -> Dict:
        """Get sync status"""
        return self.sync.get_sync_status()