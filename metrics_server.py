"""
Metrics Server - Fixed Version
Exports Prometheus metrics without middleware errors
"""

import logging
from contextlib import asynccontextmanager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI
    from fastapi.responses import Response, JSONResponse
    import uvicorn
    from prometheus_client import generate_latest, REGISTRY
    
    # Initialize metrics BEFORE creating app
    try:
        from src.prometheus_metrics_proper import init_metrics, get_prometheus_registry
        logger.info("Importing prometheus_metrics_proper...")
        init_metrics()
        registry = get_prometheus_registry()
        logger.info("✅ Prometheus metrics initialized")
    except ImportError as e:
        logger.error(f"❌ Cannot import prometheus_metrics_proper: {e}")
        logger.error("Make sure src/prometheus_metrics_proper.py exists")
        raise
    
except ImportError as e:
    logger.error(f"❌ Missing dependency: {e}")
    logger.error("Install with: pip install fastapi uvicorn prometheus-client")
    raise


# Lifespan context manager (no middleware issues)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Metrics Server starting...")
    logger.info(f"📊 Prometheus registry initialized")
    logger.info(f"🔗 Metrics endpoint: /metrics")
    logger.info(f"💚 Health check: /health")
    yield
    # Shutdown
    logger.info("⏹️ Metrics Server shutting down...")


# Create FastAPI app with lifespan (NO middleware)
app = FastAPI(
    title="RAG Metrics Server",
    description="Prometheus metrics endpoint for RAG system",
    version="2.0.0",
    lifespan=lifespan
)


# ===== ENDPOINTS =====

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "rag-metrics-server",
        "metrics_endpoint": "/metrics",
        "version": "2.0.0"
    }


@app.get("/metrics", tags=["Metrics"])
async def get_metrics():
    """
    Prometheus metrics endpoint
    Returns all RAG metrics in Prometheus format
    """
    try:
        metrics_data = generate_latest(registry)
        return Response(
            content=metrics_data,
            media_type="text/plain; charset=utf-8"
        )
    except Exception as e:
        logger.error(f"❌ Error generating metrics: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/metrics_json", tags=["Metrics"])
async def get_metrics_json():
    """Get metrics info in JSON format"""
    return {
        "status": "ok",
        "metrics_endpoint": "/metrics",
        "format": "Prometheus text format",
        "help": "Use /metrics endpoint for actual metrics"
    }


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint - API info"""
    return {
        "service": "RAG Metrics Server",
        "version": "2.0.0",
        "endpoints": {
            "/health": "Health check",
            "/metrics": "Prometheus metrics",
            "/metrics_json": "Metrics info",
            "/docs": "API documentation",
            "/redoc": "ReDoc documentation"
        }
    }


@app.get("/debug/registry", tags=["Debug"])
async def debug_registry():
    """Debug endpoint - show registry info"""
    try:
        metrics_text = generate_latest(registry).decode('utf-8')
        metrics_lines = metrics_text.split('\n')
        
        # Count different metric types
        counter_count = sum(1 for line in metrics_lines if line.startswith('rag_') and '{' in line)
        comment_count = sum(1 for line in metrics_lines if line.startswith('# HELP rag_'))
        
        return {
            "registry": "Active",
            "metrics_total": counter_count,
            "metric_types": comment_count,
            "sample_metrics": [
                line for line in metrics_lines 
                if line.startswith('rag_') and '{' in line
            ][:5]
        }
    except Exception as e:
        logger.error(f"Debug error: {e}")
        return {"error": str(e)}


# ===== ERROR HANDLERS =====

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"❌ Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "type": type(exc).__name__
        }
    )


# ===== STARTUP CHECK =====

@app.on_event("startup")
async def startup_event():
    """Verify metrics on startup"""
    try:
        metrics = generate_latest(registry)
        if b'rag_' in metrics:
            logger.info("✅ RAG metrics found in registry")
        else:
            logger.warning("⚠️  No RAG metrics in registry yet (normal on first start)")
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")


# ===== MAIN =====

def main():
    """Run metrics server"""
    logger.info("=" * 60)
    logger.info("RAG METRICS SERVER v2.0.0")
    logger.info("=" * 60)
    logger.info("")
    logger.info("📍 Starting server on http://0.0.0.0:8000")
    logger.info("")
    logger.info("Available endpoints:")
    logger.info("  • GET  /health          - Health check")
    logger.info("  • GET  /metrics         - Prometheus metrics")
    logger.info("  • GET  /metrics_json    - Metrics info")
    logger.info("  • GET  /docs            - API docs (Swagger)")
    logger.info("  • GET  /redoc           - ReDoc docs")
    logger.info("  • GET  /debug/registry  - Debug info")
    logger.info("")
    logger.info("Test with:")
    logger.info("  curl http://localhost:8000/health")
    logger.info("  curl http://localhost:8000/metrics | head -20")
    logger.info("")
    logger.info("=" * 60)
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True,
            # NO middleware_stack here - let FastAPI handle it
        )
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        raise


if __name__ == "__main__":
    main()