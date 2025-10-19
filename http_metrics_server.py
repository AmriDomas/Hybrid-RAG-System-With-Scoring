"""
Simple HTTP server for metrics - fixed version
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import time
import sys
import os
import logging

# Add path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MetricsHandler(BaseHTTPRequestHandler):
    
    def do_GET(self):
        logger.info(f"GET {self.path} from {self.client_address[0]}")
        
        if self.path == '/':
            self._send_json_response(200, {
                "service": "RAG Metrics Server", 
                "status": "running",
                "version": "2.0.0",
                "endpoints": {
                    "health": "/health",
                    "metrics": "/metrics",
                    "test": "/test",
                    "generate": "/generate-test"
                }
            })
            
        elif self.path == '/health':
            self._send_json_response(200, {
                "status": "healthy", 
                "timestamp": time.time()
            })
            
        elif self.path == '/test':
            self._handle_test()
            
        elif self.path == '/generate-test':
            self._handle_generate_test()
            
        elif self.path == '/metrics':
            self._handle_metrics()
            
        else:
            self._send_json_response(404, {"error": "Endpoint not found"})
    
    def _handle_test(self):
        """Test imports and metrics generation"""
        try:
            from src.metrics_integration import get_metrics_manager
            
            manager = get_metrics_manager()
            test_data = {
                "status": "success",
                "imports": "✅ OK",
                "manager": type(manager).__name__,
                "timestamp": time.time()
            }
            
            self._send_json_response(200, test_data)
            
        except Exception as e:
            self._send_json_response(500, {
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            })
    
    def _handle_generate_test(self):
        """Generate test metrics dengan evaluation data yang lengkap - CONSISTENT"""
        try:
            from src.metrics_integration import get_metrics_manager
            
            manager = get_metrics_manager()
            
            # Generate test metrics dengan evaluation data
            test_queries = [
                {
                    "method": "hybrid", 
                    "latency": 1200, 
                    "tokens": 450, 
                    "docs": 3,
                    "answer_length": 250,
                    "faithfulness": 0.92,
                    "relevance": 0.88,
                    "correctness": 0.85,
                    "hallucination_rate": 0.05,
                    "overall_score": 0.88,
                    "recall_at_5": 0.80,
                    "precision_at_5": 0.75,
                    "mrr": 0.70
                },
                {
                    "method": "semantic", 
                    "latency": 800, 
                    "tokens": 320, 
                    "docs": 2,
                    "answer_length": 180,
                    "faithfulness": 0.95,
                    "relevance": 0.90, 
                    "correctness": 0.92,
                    "hallucination_rate": 0.03,
                    "overall_score": 0.92,
                    "recall_at_5": 0.85,
                    "precision_at_5": 0.80,
                    "mrr": 0.75
                },
            ]
            
            for i, query in enumerate(test_queries):
                metrics_data = {
                    'faithfulness': query['faithfulness'],
                    'relevance': query['relevance'],
                    'correctness': query['correctness'],
                    'hallucination_rate': query['hallucination_rate'],
                    'overall_score': query['overall_score'],
                    'retrieval': {
                        'recall@5': query['recall_at_5'],
                        'precision@5': query['precision_at_5'],
                        'mrr': query['mrr']
                    },
                    'evaluation_enabled': True
                }
                
                # GUNAKAN PARAMETER YANG KONSISTEN
                manager.record_query_metrics(
                    retrieval_method=query["method"],
                    latency_ms=query["latency"],
                    tokens=query["tokens"],
                    num_documents=query["docs"],  # PARAMETER YANG KONSISTEN
                    answer_length=query["answer_length"],
                    metrics=metrics_data,
                    status='success',
                    has_evaluation=True
                )
                logger.info(f"Generated test metric: {query['method']} with evaluation")
            
            self._send_json_response(200, {
                "status": "success",
                "message": f"Generated {len(test_queries)} test metrics with evaluation data",
                "test_queries": test_queries,
                "timestamp": time.time()
            })
            
        except Exception as e:
            logger.error(f"Error in generate-test: {e}")
            self._send_json_response(500, {
                "status": "error", 
                "error": str(e),
                "traceback": __import__('traceback').format_exc(),
                "timestamp": time.time()
            })
    
    def _handle_metrics(self):
        """Serve metrics in Prometheus format"""
        try:
            from src.metrics_integration import get_metrics_manager
            
            manager = get_metrics_manager()
            metrics_data = manager.get_all_metrics_prometheus_format()
            
            if metrics_data and len(metrics_data) > 0:
                content = metrics_data.decode('utf-8')
                logger.info(f"Serving {len(content)} bytes of metrics data")
                
                self.send_response(200)
                self.send_header('Content-type', 'text/plain')
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content.encode())
            else:
                self._send_text_response(200, "# No metrics data available\n")
                
        except Exception as e:
            error_msg = f"# Error generating metrics: {str(e)}\n"
            logger.error(f"Metrics error: {e}")
            self._send_text_response(500, error_msg)
    
    def _send_json_response(self, code, data):
        """Send JSON response"""
        response = json.dumps(data, indent=2).encode()
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Content-Length', str(len(response)))
        self.end_headers()
        self.wfile.write(response)
    
    def _send_text_response(self, code, text):
        """Send text response"""
        response = text.encode()
        self.send_response(code)
        self.send_header('Content-type', 'text/plain')
        self.send_header('Content-Length', str(len(response)))
        self.end_headers()
        self.wfile.write(response)
    
    def log_message(self, format, *args):
        # Use our logger instead of default
        logger.info("%s - - [%s] %s" % (self.client_address[0], self.log_date_time_string(), format % args))

def run_server(port=8003):
    """Run the HTTP server"""
    server = HTTPServer(('0.0.0.0', port), MetricsHandler)
    
    print("=" * 50)
    print(f"🚀 Starting Simple HTTP Metrics Server")
    print(f"📍 http://localhost:{port}")
    print("=" * 50)
    
    # Test imports before starting
    try:
        from src.metrics_integration import get_metrics_manager
        manager = get_metrics_manager()
        print("✅ All imports successful")
        
        # Generate initial test data menggunakan parameter yang KONSISTEN
        manager.record_query_metrics(
            retrieval_method="initial",
            latency_ms=1000,
            tokens=200,
            num_documents=5,  # PARAMETER YANG DITAMBAHKAN
            answer_length=150,  # PARAMETER YANG DITAMBAHKAN
            metrics={
                'documents_retrieved': 5,
                'faithfulness': 0.9,
                'relevance': 0.85,
                'evaluation_enabled': True
            },
            status='success',
            has_evaluation=True
        )
        print("✅ Initial test data generated")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("📊 Server ready! Endpoints:")
    print("   GET /          - Server info")
    print("   GET /health    - Health check") 
    print("   GET /test      - Test imports")
    print("   GET /generate-test - Generate test metrics")
    print("   GET /metrics   - Prometheus metrics")
    print("=" * 50)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
    finally:
        server.server_close()

if __name__ == "__main__":
    run_server()