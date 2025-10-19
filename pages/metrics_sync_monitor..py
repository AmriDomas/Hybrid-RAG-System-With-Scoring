"""
Streamlit page for monitoring metrics sync between app and Prometheus
Complete fixed version
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.metrics_integration import get_metrics_manager
except ImportError:
    st.error("Error: Could not import metrics_integration module")
    st.stop()

st.set_page_config(
    page_title="Metrics Sync Monitor",
    page_icon="🔄",
    layout="wide"
)

st.title("🔄 Metrics Synchronization Monitor")
st.markdown("Monitors sync between app.py and Prometheus")
st.markdown("---")

# Get metrics manager
try:
    manager = get_metrics_manager()
except Exception as e:
    st.error(f"Error initializing metrics manager: {str(e)}")
    st.info("Make sure metrics_integration.py and metrics_sync.py are in src/ folder")
    st.stop()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔄 Sync Status",
    "✅ Validation",
    "📊 Buffer",
    "🔧 Debug"
])

# ===== TAB 1: Sync Status =====
with tab1:
    st.header("Synchronization Status")
    
    try:
        sync_status = manager.get_sync_status()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_emoji = {
                'fresh': '🟢',
                'ok': '🟡',
                'stale': '🔴',
                'no_data': '⚫'
            }
            emoji = status_emoji.get(sync_status.get('status', 'unknown'), '❓')
            
            st.metric(
                "Sync Status",
                f"{emoji} {sync_status.get('status', 'unknown').upper()}",
                help="fresh=<5s, ok=5-30s, stale=>30s"
            )
        
        with col2:
            st.metric(
                "Buffered Metrics",
                sync_status.get('buffered_metrics', 0),
                help="Number of metric snapshots in buffer"
            )
        
        with col3:
            age_seconds = sync_status.get('age_seconds', 0)
            st.metric(
                "Age (seconds)",
                f"{age_seconds:.1f}s",
                help="How old the latest metrics are"
            )
        
        with col4:
            syncing = "✅ ON" if sync_status.get('syncing', False) else "❌ OFF"
            st.metric(
                "Sync Service",
                syncing
            )
        
        st.markdown("---")
        
        # Last update
        col1, col2 = st.columns(2)
        
        with col1:
            if sync_status.get('last_update'):
                st.markdown(f"**Last Update:** {sync_status['last_update']}")
            else:
                st.markdown("**Last Update:** No data yet")
        
        with col2:
            if st.button("🔄 Force Sync Now"):
                with st.spinner("Forcing sync..."):
                    success = manager.force_sync()
                    if success:
                        st.success("✅ Sync forced successfully!")
                    else:
                        st.error("❌ Force sync failed")
        
        st.markdown("---")
        
        # Prometheus info
        st.subheader("ℹ️ About Metrics Sync")
        
        st.markdown("""
        **How it works:**
        1. App.py records metrics → Stored in buffer
        2. Sync service monitors buffer (every 5s)
        3. Prometheus scrapes metrics (every 15s by default)
        4. If metrics are fresh, callbacks trigger
        5. Metrics exported to Prometheus endpoint
        
        **Status Meanings:**
        - 🟢 **Fresh**: Metrics < 5 seconds old (optimal)
        - 🟡 **OK**: Metrics 5-30 seconds old (acceptable)
        - 🔴 **Stale**: Metrics > 30 seconds old (issue)
        - ⚫ **No Data**: No metrics recorded yet
        
        **Why sync matters:**
        - Prometheus dashboard stays current
        - Grafana shows real-time data
        - No confusion between app and dashboard
        - Helps debug metric issues
        """)
    
    except Exception as e:
        st.error(f"Error in Sync Status tab: {str(e)}")
        st.info("Debugging info: " + str(type(e).__name__))

# ===== TAB 2: Validation =====
with tab2:
    st.header("Metrics Validation")
    
    try:
        recon_status = manager.get_reconciliation_status()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Recent Checks",
                recon_status.get('recent_checks', 0)
            )
        
        with col2:
            synced_count = recon_status.get('synced', 0)
            st.metric(
                "In Sync",
                synced_count,
                f"+{synced_count} out of {recon_status.get('recent_checks', 1)}"
            )
        
        with col3:
            sync_rate = recon_status.get('sync_rate', '0%')
            st.metric(
                "Sync Rate",
                sync_rate,
                delta=sync_rate
            )
        
        st.markdown("---")
        
        # Reconciliation logs
        st.subheader("Recent Reconciliation Logs")
        
        logs = manager.reconciler.get_reconciliation_log(limit=10)
        
        if logs:
            for i, log in enumerate(reversed(logs[-5:]), 1):
                log_num = len(logs) - 5 + i
                is_first = (i == 1)
                
                with st.expander(
                    f"Check #{log_num} - {log.get('timestamp', 'Unknown')}", 
                    expanded=is_first
                ):
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if log.get('synced'):
                            st.success("✅ SYNCED")
                        else:
                            st.error("❌ OUT OF SYNC")
                    
                    with col2:
                        st.caption(log.get('timestamp', 'No timestamp'))
                    
                    if log.get('differences'):
                        st.markdown("**Differences:**")
                        for diff in log['differences']:
                            st.markdown(f"- {diff}")
                    
                    if log.get('missing_in_prometheus'):
                        st.markdown("**Missing in Prometheus:**")
                        for metric in log['missing_in_prometheus']:
                            st.markdown(f"- {metric}")
                    
                    if log.get('extra_in_prometheus'):
                        st.markdown("**Extra in Prometheus:**")
                        for metric in log['extra_in_prometheus']:
                            st.markdown(f"- {metric}")
        else:
            st.info("No validation logs yet")
    
    except Exception as e:
        st.error(f"Error in Validation tab: {str(e)}")

# ===== TAB 3: Buffer =====
with tab3:
    st.header("Metrics Buffer")
    
    try:
        current_metrics = manager.synced_collector.get_current_metrics()
        
        if current_metrics:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**Current Buffered Metrics:**")
                
                metrics_display = []
                for key, value in current_metrics.items():
                    if isinstance(value, float):
                        value_str = f"{value:.4f}"
                    else:
                        value_str = str(value)
                    
                    metrics_display.append({
                        'Metric': key,
                        'Value': value_str
                    })
                
                if metrics_display:
                    st.dataframe(
                        pd.DataFrame(metrics_display),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No metrics in display")
            
            with col2:
                st.metric(
                    "Total Metrics",
                    len(current_metrics)
                )
            
            st.markdown("---")
            
            # Export buffer
            if st.button("📥 Export Current Buffer"):
                json_str = json.dumps(current_metrics, indent=2, default=str)
                st.download_button(
                    "Download Metrics JSON",
                    json_str,
                    "metrics_buffer.json",
                    "application/json"
                )
            
            # Show raw data
            with st.expander("📋 View Raw JSON"):
                st.json(current_metrics)
        else:
            st.info("No metrics in buffer yet. Metrics appear after first query.")
    
    except Exception as e:
        st.error(f"Error in Buffer tab: {str(e)}")

# ===== TAB 4: Debug =====
with tab4:
    st.header("🔧 Debug Information")
    
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sync Service Info")
            
            sync_status = manager.get_sync_status()
            debug_info_1 = {
                'service_running': sync_status.get('syncing', False),
                'buffer_size': sync_status.get('buffered_metrics', 0),
                'last_update': sync_status.get('last_update', 'None'),
                'age_seconds': round(sync_status.get('age_seconds', 0), 2)
            }
            st.json(debug_info_1)
        
        with col2:
            st.subheader("Validation Info")
            
            recon_status = manager.get_reconciliation_status()
            debug_info_2 = {
                'recent_checks': recon_status.get('recent_checks', 0),
                'synced': recon_status.get('synced', 0),
                'out_of_sync': recon_status.get('out_of_sync', 0),
                'sync_rate': recon_status.get('sync_rate', '0%')
            }
            st.json(debug_info_2)
        
        st.markdown("---")
        
        # Configuration info
        st.subheader("Configuration")
        
        st.markdown(f"""
        **Sync Configuration:**
        - Prometheus Scrape Interval: 15 seconds
        - App Update Interval: 5 seconds
        - Sync Check Interval: 5 seconds
        - Buffer Size: 200 snapshots
        - Buffer Retention: 300 seconds (5 minutes)
        
        **Expected Behavior:**
        - App records metrics immediately
        - Prometheus scrapes every 15 seconds
        - Metrics should be in sync within 15-20 seconds
        - Buffer keeps last 5 minutes of metrics
        """)
        
        st.markdown("---")
        
        # Connection test
        st.subheader("Connection Tests")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧪 Test Prometheus Connection"):
                try:
                    import requests
                    with st.spinner("Testing connection..."):
                        response = requests.get(
                            "http://localhost:9090/-/healthy", 
                            timeout=5
                        )
                        if response.status_code == 200:
                            st.success("✅ Prometheus is reachable")
                        else:
                            st.error(f"❌ Prometheus returned {response.status_code}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to Prometheus (connection refused)")
                except requests.exceptions.Timeout:
                    st.error("❌ Prometheus connection timeout")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        with col2:
            if st.button("🧪 Test Metrics Export"):
                try:
                    with st.spinner("Testing metrics export..."):
                        metrics_bytes = manager.get_all_metrics_prometheus_format()
                        metric_lines = metrics_bytes.decode('utf-8').split('\n')
                        active_metrics = [
                            l for l in metric_lines 
                            if not l.startswith('#') and l.strip()
                        ]
                        st.success(f"✅ Exporting {len(active_metrics)} active metrics")
                        
                        with st.expander("View first 10 metrics"):
                            for line in active_metrics[:10]:
                                st.code(line, language="text")
                except Exception as e:
                    st.error(f"❌ Export failed: {str(e)}")
        
        st.markdown("---")
        
        # Troubleshooting
        st.subheader("🔍 Troubleshooting")
        
        with st.expander("Metrics not syncing?"):
            st.markdown("""
            **Check:**
            1. Is app recording metrics? (Check Sync Status tab)
            2. Is Prometheus running? (Try connection test)
            3. Is scrape interval set correctly in prometheus.yml?
            4. Are there any errors in logs?
            
            **Solutions:**
            - Wait 15-20 seconds for next scrape
            - Force sync using "Force Sync Now" button
            - Check Prometheus scrape endpoint: http://localhost:9090/metrics
            - Verify prometheus.yml has correct target
            """)
        
        with st.expander("Metrics are stale?"):
            st.markdown("""
            **Possible causes:**
            - App not recording new metrics
            - Network delay between app and Prometheus
            - Prometheus scrape interval too long
            
            **Solutions:**
            - Make a new query to generate fresh metrics
            - Check app is running and healthy
            - Reduce Prometheus scrape_interval in prometheus.yml
            - Check network connectivity
            """)
        
        with st.expander("Sync rate not 100%?"):
            st.markdown("""
            **Possible causes:**
            - Timing differences (normal)
            - Metric value changes between app and Prometheus
            - Rounding differences in floating point
            
            **Solutions:**
            - Small sync rate < 100% is normal
            - Focus on recent checks (last 10)
            - Check if differences are within tolerance
            - Review reconciliation logs for details
            """)
    
    except Exception as e:
        st.error(f"Error in Debug tab: {str(e)}")

st.markdown("---")
st.caption("🔄 Metrics Sync Monitor | Real-time sync tracking between app and Prometheus")