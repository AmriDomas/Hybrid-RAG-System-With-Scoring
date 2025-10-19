"""
Streamlit page for AI Agent Dashboard
Monitoring and control for optimization agent
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="AI Agent Dashboard",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Optimization Agent Dashboard")
st.markdown("---")

# Initialize agent manager from session
if 'agent_manager' not in st.session_state:
    st.warning("⚠️ Agent not initialized. Initialize from main app first.")
    st.stop()

agent_manager = st.session_state.agent_manager

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Status",
    "🔧 Recommendations",
    "📈 History",
    "⚙️ Control",
    "📥 Reports"
])

# ===== TAB 1: Status =====
with tab1:
    st.header("Agent Status & Statistics")
    
    status = agent_manager.get_agent_status()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Status",
            "🟢 Running" if status['is_running'] else "🔴 Stopped"
        )
    
    with col2:
        st.metric(
            "Total Optimizations",
            status['stats']['total_optimizations']
        )
    
    with col3:
        st.metric(
            "Successful Improvements",
            status['stats']['successful_improvements']
        )
    
    with col4:
        total_improvement = status['stats']['total_score_improvement']
        st.metric(
            "Total Score Gain",
            f"+{total_improvement:.1%}",
            delta=f"{total_improvement:.1%}" if total_improvement > 0 else None,
            delta_color="normal" if total_improvement > 0 else "off"
        )
    
    st.markdown("---")
    
    # Agent Metrics
    st.subheader("Current Metrics")
    
    if 'metrics' in status['agent_status']:
        metrics = status['agent_status']['metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Faithfulness",
                f"{metrics.get('faithfulness', 0):.2%}",
                help="How well answer is supported by context"
            )
        
        with col2:
            st.metric(
                "Relevance",
                f"{metrics.get('relevance', 0):.2%}",
                help="How well answer addresses question"
            )
        
        with col3:
            st.metric(
                "Hallucination Rate",
                f"{metrics.get('hallucination_rate', 0):.2%}",
                delta=f"-{metrics.get('hallucination_rate', 0):.2%}",
                delta_color="inverse",
                help="% of unsupported claims"
            )
        
        with col4:
            latency = metrics.get('latency_ms', 0)
            st.metric(
                "Latency",
                f"{latency:.0f}ms",
                help="Query response time"
            )
    
    st.markdown("---")
    
    # Current Configuration
    st.subheader("Current Configuration")
    
    config = status['agent_status'].get('current_config', {})
    
    config_df = pd.DataFrame([
        {
            'Parameter': 'Top-K Documents',
            'Value': config.get('top_k', 'N/A'),
            'Purpose': 'Number of documents to retrieve'
        },
        {
            'Parameter': 'Semantic Weight (α)',
            'Value': f"{config.get('hybrid_alpha', 0):.2f}",
            'Purpose': 'Semantic vs keyword balance'
        },
        {
            'Parameter': 'Retrieval Method',
            'Value': config.get('retrieval_method', 'hybrid'),
            'Purpose': 'Retrieval strategy'
        },
        {
            'Parameter': 'Temperature',
            'Value': f"{config.get('temperature', 0.7):.2f}",
            'Purpose': 'LLM creativity level'
        },
        {
            'Parameter': 'Max Tokens',
            'Value': config.get('max_tokens', 2000),
            'Purpose': 'Max response length'
        },
    ])
    
    st.dataframe(config_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Trends
    st.subheader("Metrics Trends")
    
    if 'metrics_trend' in status['agent_status']:
        trends = status['agent_status']['metrics_trend']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            trend, change = trends.get('faithfulness', ('stable', 0))
            emoji = "📈" if trend == "improving" else "📉" if trend == "degrading" else "➡️"
            st.markdown(f"""
            **Faithfulness:** {emoji}
            - Trend: {trend}
            - Change: {change:+.1f}%
            """)
        
        with col2:
            trend, change = trends.get('relevance', ('stable', 0))
            emoji = "📈" if trend == "improving" else "📉" if trend == "degrading" else "➡️"
            st.markdown(f"""
            **Relevance:** {emoji}
            - Trend: {trend}
            - Change: {change:+.1f}%
            """)
        
        with col3:
            trend, change = trends.get('hallucination', ('stable', 0))
            emoji = "📈" if trend == "degrading" else "📉" if trend == "improving" else "➡️"
            st.markdown(f"""
            **Hallucination:** {emoji}
            - Trend: {trend}
            - Change: {change:+.1f}%
            """)

# ===== TAB 2: Recommendations =====
with tab2:
    st.header("Recommended Optimizations")
    
    recommendations = agent_manager.recommend_actions()
    
    if not recommendations:
        st.success("✅ System is already optimized!")
    else:
        for i, rec in enumerate(recommendations, 1):
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**#{i} {rec['action'].upper()}**")
                    st.markdown(f"_{rec['description']}_")
                    st.caption(f"Parameters: {rec['parameters']}")
                
                with col2:
                    if st.button("🚀 Apply", key=f"apply_{i}"):
                        success = agent_manager.apply_recommendation(
                            rec['action'],
                            rec['parameters']
                        )
                        if success:
                            st.success("✅ Applied!")
                        else:
                            st.error("❌ Failed to apply")
                
                st.divider()

# ===== TAB 3: History =====
with tab3:
    st.header("Optimization History")
    
    history = agent_manager.get_optimization_history(limit=20)
    
    if not history:
        st.info("No optimization history yet.")
    else:
        # Summary stats from history
        improvements = [h.get('improvement', 0) for h in history]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Optimizations Run",
                len(history)
            )
        
        with col2:
            positive = sum(1 for imp in improvements if imp > 0)
            st.metric(
                "Successful Cycles",
                positive
            )
        
        with col3:
            avg_improvement = sum(improvements) / len(improvements) if improvements else 0
            st.metric(
                "Avg Improvement",
                f"+{avg_improvement:.2%}"
            )
        
        st.markdown("---")
        
        # Improvement over time
        st.subheader("Score Improvement Trend")
        
        history_df = pd.DataFrame([
            {
                'Cycle': i + 1,
                'Improvement': h.get('improvement', 0),
                'Final Score': h['iterations'][-1]['score'] if h.get('iterations') else 0
            }
            for i, h in enumerate(history)
        ])
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=history_df['Cycle'],
            y=history_df['Improvement'],
            name='Improvement',
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Scatter(
            x=history_df['Cycle'],
            y=history_df['Final Score'],
            name='Final Score',
            yaxis='y2',
            line=dict(color='red'),
            mode='lines+markers'
        ))
        
        fig.update_layout(
            title="Optimization Cycles",
            xaxis_title="Cycle",
            yaxis_title="Score Improvement",
            yaxis2=dict(title="Final Score", overlaying='y', side='right'),
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Recent optimizations detail
        st.subheader("Recent Optimization Details")
        
        for i, h in enumerate(reversed(history[-5:]), 1):
            with st.expander(f"📊 Cycle {len(history) - 5 + i}"):
                cols_info = st.columns(2)
                
                with cols_info[0]:
                    st.markdown("**Actions Taken:**")
                    for iteration in h.get('iterations', [])[1:]:
                        if iteration.get('action'):
                            st.markdown(f"- {iteration['action']}")
                
                with cols_info[1]:
                    st.markdown("**Result:**")
                    st.markdown(f"- Improvement: {h.get('improvement', 0):.2%}")
                    st.markdown(f"- Final Score: {h['iterations'][-1]['score']:.2%}")
                
                # Metrics comparison
                st.markdown("**Metrics Comparison:**")
                
                initial = h['iterations'][0]['metrics']
                final = h['iterations'][-1]['metrics']
                
                comparison = pd.DataFrame([
                    {
                        'Metric': 'Faithfulness',
                        'Initial': f"{initial.get('faithfulness', 0):.2%}",
                        'Final': f"{final.get('faithfulness', 0):.2%}",
                        'Change': f"{final.get('faithfulness', 0) - initial.get('faithfulness', 0):+.2%}"
                    },
                    {
                        'Metric': 'Relevance',
                        'Initial': f"{initial.get('relevance', 0):.2%}",
                        'Final': f"{final.get('relevance', 0):.2%}",
                        'Change': f"{final.get('relevance', 0) - initial.get('relevance', 0):+.2%}"
                    },
                    {
                        'Metric': 'Hallucination',
                        'Initial': f"{initial.get('hallucination_rate', 0):.2%}",
                        'Final': f"{final.get('hallucination_rate', 0):.2%}",
                        'Change': f"{final.get('hallucination_rate', 0) - initial.get('hallucination_rate', 0):+.2%}"
                    },
                ])
                
                st.dataframe(comparison, use_container_width=True, hide_index=True)

# ===== TAB 4: Control =====
with tab4:
    st.header("Agent Control")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Manual Optimization")
        
        if st.button("▶️ Run Single Optimization Cycle"):
            with st.spinner("Running optimization..."):
                report = agent_manager.run_single_optimization(
                    test_query="What is machine learning?",
                    max_iterations=3
                )
                
                if 'error' not in report:
                    st.success(f"✅ Optimization complete!")
                    st.metric(
                        "Score Improvement",
                        f"+{report.get('improvement', 0):.2%}"
                    )
                    
                    with st.expander("📊 Detailed Report"):
                        st.json(report)
                else:
                    st.error(f"❌ Error: {report['error']}")
    
    with col2:
        st.subheader("Continuous Monitoring")
        
        monitoring_status = st.session_state.get('agent_monitoring_status', False)
        
        if not monitoring_status:
            if st.button("▶️ Start Continuous Monitoring"):
                st.session_state.agent_monitoring_status = True
                st.success("✅ Continuous monitoring started!")
                st.info("Agent will optimize system every 5 minutes in the background")
        else:
            if st.button("⏹️ Stop Continuous Monitoring"):
                st.session_state.agent_monitoring_status = False
                st.info("Continuous monitoring stopped")

# ===== TAB 5: Reports =====
with tab5:
    st.header("Export & Reports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Generate Comprehensive Report"):
            filepath = f"agent_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            success = agent_manager.export_report(filepath)
            
            if success:
                st.success(f"✅ Report generated: {filepath}")
                
                with open(filepath, 'r') as f:
                    report_data = json.load(f)
                
                st.download_button(
                    "📥 Download Report",
                    json.dumps(report_data, indent=2),
                    filepath,
                    "application/json"
                )
    
    with col2:
        st.info("Reports include:")
        st.markdown("""
        - Current agent status
        - Recommended optimizations
        - Optimization history (50 recent)
        - Current system configuration
        - Metrics trends
        """)

st.markdown("---")
st.caption("🤖 AI Optimization Agent Dashboard | Auto-refresh enabled")