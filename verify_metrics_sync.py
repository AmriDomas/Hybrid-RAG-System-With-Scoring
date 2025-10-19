"""
Script untuk verifikasi sinkronisasi metrik
"""

import requests
import json
import time

def check_prometheus_metrics():
    """Check if RAG metrics are in Prometheus"""
    try:
        response = requests.get("http://localhost:9090/api/v1/query?query=rag_queries_total")
        data = response.json()
        
        if data['data']['result']:
            print("✅ RAG metrics found in Prometheus")
            for result in data['data']['result']:
                print(f"  - {result['metric']['__name__']}: {result['value'][1]}")
        else:
            print("❌ No RAG metrics in Prometheus")
            
    except Exception as e:
        print(f"❌ Error checking Prometheus: {e}")

def check_metrics_server():
    """Check metrics server endpoint"""
    try:
        response = requests.get("http://localhost:8000/metrics")
        if "rag_queries_total" in response.text:
            print("✅ RAG metrics available from metrics server")
        else:
            print("❌ RAG metrics not in metrics server")
            
    except Exception as e:
        print(f"❌ Error checking metrics server: {e}")

def check_grafana_dashboard():
    """Verify Grafana can access metrics"""
    try:
        # This would require Grafana API, but we can check connectivity
        response = requests.get("http://localhost:3000", timeout=5)
        print("✅ Grafana is accessible")
    except:
        print("❌ Grafana not accessible")

if __name__ == "__main__":
    print("🔍 Verifying Metrics Synchronization...")
    print("=" * 50)
    
    check_prometheus_metrics()
    check_metrics_server() 
    check_grafana_dashboard()
    
    print("=" * 50)
    print("📊 Next steps:")
    print("1. Run: python metrics_server.py")
    print("2. Run: streamlit run app.py") 
    print("3. Make some queries in the app")
    print("4. Check: http://localhost:8000/metrics")
    print("5. Check: http://localhost:9090/graph")
    print("6. View dashboard: http://localhost:3000")