"""
Verify full metrics integration
"""

import requests
import time
import json

def test_integration():
    print("🔍 Verifying Full Metrics Integration")
    print("=" * 60)
    
    # Test 1: Metrics Server
    print("\n1. 📊 Testing Metrics Server...")
    try:
        response = requests.get("http://localhost:8003/metrics", timeout=5)
        if response.status_code == 200:
            content = response.text
            rag_metrics = [line for line in content.split('\n') if 'rag_' in line and not line.startswith('#')]
            print(f"   ✅ Metrics Server: {len(rag_metrics)} RAG metrics found")
        else:
            print(f"   ❌ Metrics Server: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Metrics Server: {e}")
    
    # Test 2: Prometheus
    print("\n2. 📈 Testing Prometheus...")
    try:
        response = requests.get("http://localhost:9090/api/v1/query?query=rag_queries_total", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data['data']['result']:
                print(f"   ✅ Prometheus: {len(data['data']['result'])} RAG metrics found")
                for result in data['data']['result']:
                    print(f"      - {result['metric']['retrieval_method']}: {result['value'][1]}")
            else:
                print("   ⚠️  Prometheus: No RAG metrics yet (wait for scrape)")
        else:
            print(f"   ❌ Prometheus: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Prometheus: {e}")
    
    # Test 3: Streamlit App (if running)
    print("\n3. 🎯 Testing Streamlit App...")
    try:
        response = requests.get("http://localhost:8501", timeout=5)
        if response.status_code == 200:
            print("   ✅ Streamlit: Running")
        else:
            print(f"   ⚠️  Streamlit: {response.status_code}")
    except Exception as e:
        print(f"   ⚠️  Streamlit: {e} (might not be running)")
    
    # Test 4: Generate test traffic
    print("\n4. 🚀 Generating Test Traffic...")
    try:
        response = requests.get("http://localhost:8003/generate-test", timeout=5)
        if response.status_code == 200:
            print("   ✅ Test metrics generated")
            time.sleep(2)  # Wait for Prometheus scrape
            
            # Check if metrics updated in Prometheus
            response = requests.get("http://localhost:9090/api/v1/query?query=rag_queries_total", timeout=5)
            data = response.json()
            total_queries = sum(float(result['value'][1]) for result in data['data']['result'])
            print(f"   📈 Total queries in Prometheus: {total_queries}")
        else:
            print(f"   ❌ Test generation failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Test generation: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Integration Test Complete!")
    print("\nNext steps:")
    print("1. Open Streamlit app and make real queries")
    print("2. Check Prometheus: http://localhost:9090")
    print("3. Setup Grafana dashboard with the provided JSON")
    print("4. Monitor real-time metrics!")

if __name__ == "__main__":
    test_integration()