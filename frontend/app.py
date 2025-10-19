import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import io
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vector_store import VectorStore
from src.llm_client import LLMClient
from src.rag_engine import RAGEngine
from src.metrics_integration import record_rag_query, get_metrics_manager

# Import metrics yang compatible
try:
    from monitoring.metrics import (
        custom_registry,
        MetricsCollector,
        monitor_query,
        query_counter,
        total_latency,
        documents_retrieved,
        indexed_documents
    )
    from prometheus_client import generate_latest
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    logger.warning("Metrics module not available, continuing without advanced metrics")

st.set_page_config(
    page_title="Hybrid RAG System - Multi-File",
    page_icon="🤖",
    layout="wide"
)

def metrics():
    if METRICS_AVAILABLE:
        return generate_latest(custom_registry)
    return b""

# Session state initialization
if 'history' not in st.session_state:
    st.session_state.history = []
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'agent_manager' not in st.session_state:
    st.session_state.agent_manager = None
if 'enable_evaluation' not in st.session_state:
    st.session_state.enable_evaluation = False
if 'use_llm_judge' not in st.session_state:
    st.session_state.use_llm_judge = False
if 'agent_monitoring_status' not in st.session_state:
    st.session_state.agent_monitoring_status = False

def parse_uploaded_file(uploaded_file):
    """Parse uploaded file based on type"""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    documents = []
    
    try:
        if file_extension == 'txt':
            content = uploaded_file.read().decode('utf-8')
            documents.append({
                'id': f"upload_{uploaded_file.name}_{datetime.now().timestamp()}",
                'title': uploaded_file.name,
                'content': content,
                'category': 'Uploaded',
                'metadata': {
                    'source': uploaded_file.name,
                    'upload_date': datetime.now().isoformat(),
                    'file_type': 'text',
                    'tags': ['uploaded']
                }
            })
        
        elif file_extension == 'json':
            content = json.load(uploaded_file)
            if isinstance(content, list):
                for idx, item in enumerate(content):
                    if isinstance(item, dict) and 'content' in item:
                        doc = {
                            'id': item.get('id', f"upload_{uploaded_file.name}_{idx}"),
                            'title': item.get('title', f"Document {idx+1}"),
                            'content': item['content'],
                            'category': item.get('category', 'Uploaded'),
                            'metadata': item.get('metadata', {
                                'source': uploaded_file.name,
                                'upload_date': datetime.now().isoformat(),
                                'tags': ['uploaded']
                            })
                        }
                        documents.append(doc)
        
        elif file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
            for idx, row in df.iterrows():
                content_col = None
                for col in ['content', 'text', 'description', 'body']:
                    if col in df.columns:
                        content_col = col
                        break
                
                if content_col:
                    documents.append({
                        'id': f"csv_{uploaded_file.name}_{idx}",
                        'title': row.get('title', f"Row {idx+1}"),
                        'content': str(row[content_col]),
                        'category': row.get('category', 'CSV Upload'),
                        'metadata': {
                            'source': uploaded_file.name,
                            'row_index': idx,
                            'upload_date': datetime.now().isoformat(),
                            'tags': ['uploaded', 'csv']
                        }
                    })
        
        elif file_extension in ['xlsx', 'xls']:
            try:
                import openpyxl
                df = pd.read_excel(uploaded_file)
                for idx, row in df.iterrows():
                    content_col = None
                    for col in ['content', 'text', 'description', 'body']:
                        if col in df.columns:
                            content_col = col
                            break
                    
                    if content_col:
                        documents.append({
                            'id': f"excel_{uploaded_file.name}_{idx}",
                            'title': row.get('title', f"Row {idx+1}"),
                            'content': str(row[content_col]),
                            'category': row.get('category', 'Excel Upload'),
                            'metadata': {
                                'source': uploaded_file.name,
                                'row_index': idx,
                                'upload_date': datetime.now().isoformat(),
                                'tags': ['uploaded', 'excel']
                            }
                        })
            except ImportError:
                st.error("openpyxl not installed. Install with: pip install openpyxl")
        
        elif file_extension == 'pdf':
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                full_text = ""
                for page in pdf_reader.pages:
                    full_text += page.extract_text()
                
                documents.append({
                    'id': f"pdf_{uploaded_file.name}_{datetime.now().timestamp()}",
                    'title': uploaded_file.name,
                    'content': full_text,
                    'category': 'PDF Upload',
                    'metadata': {
                        'source': uploaded_file.name,
                        'pages': len(pdf_reader.pages),
                        'upload_date': datetime.now().isoformat(),
                        'tags': ['uploaded', 'pdf']
                    }
                })
            except ImportError:
                st.error("PyPDF2 not installed. Install with: pip install PyPDF2")
    
    except Exception as e:
        st.error(f"Error parsing {uploaded_file.name}: {str(e)}")
    
    return documents

def initialize_rag_with_data(documents, api_key):
    """Initialize RAG system with uploaded documents"""
    try:
        vector_store = VectorStore()
        
        # Load documents directly
        vector_store.documents = documents
        texts = [doc['content'] for doc in documents]
        vector_store.embeddings = vector_store.encoder.encode(texts, show_progress_bar=True)
        
        # Build FAISS index
        import faiss
        import numpy as np
        dimension = vector_store.embeddings.shape[1]
        vector_store.index = faiss.IndexFlatL2(dimension)
        vector_store.index.add(vector_store.embeddings.astype('float32'))
        
        # Initialize LLM with API key
        llm_client = LLMClient(api_key=api_key, model="gpt-4", temperature=0.7)
        rag_engine = RAGEngine(vector_store, llm_client, top_k=3)
        
        # Set metrics jika available
        if METRICS_AVAILABLE:
            MetricsCollector.set_indexed_documents(len(documents))
            MetricsCollector.set_system_info({
                'version': '2.0.0',
                'embedding_model': 'all-MiniLM-L6-v2',
                'llm_model': 'gpt-4'
            })
        
        return rag_engine
    except Exception as e:
        st.error(f"Error initializing RAG: {str(e)}")
        return None

# Main UI
st.title("🤖 Hybrid RAG System - Multi-File Upload")
st.markdown("---")

# Sidebar - Configuration
with st.sidebar:
    
    st.header("🔑 API Configuration")
    
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.api_key or "",
        help="Enter your OpenAI API key (required)"
    )
    
    if api_key_input and api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input
        st.success("✅ API Key saved!")
    
    st.markdown("---")
    st.header("📁 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Upload multiple files",
        type=['txt', 'json', 'csv', 'xlsx', 'xls', 'pdf'],
        accept_multiple_files=True,
        help="Supported: TXT, JSON, CSV, Excel, PDF"
    )
    
    if uploaded_files:
        st.info(f"📄 {len(uploaded_files)} file(s) selected")
        
        if st.button("🚀 Process Files", type="primary"):
            if not st.session_state.api_key:
                st.error("⚠️ Please enter API Key first!")
            else:
                with st.spinner("Processing uploaded files..."):
                    all_docs = []
                    for file in uploaded_files:
                        docs = parse_uploaded_file(file)
                        all_docs.extend(docs)
                        st.session_state.uploaded_files.append(file.name)
                    
                    if all_docs:
                        st.session_state.documents = all_docs
                        st.session_state.rag_engine = initialize_rag_with_data(
                            all_docs,
                            st.session_state.api_key
                        )
                        st.success(f"✅ Processed {len(all_docs)} documents!")
                    else:
                        st.error("No valid documents found in uploaded files")
    
    st.markdown("---")
    
    if st.session_state.rag_engine:
        st.header("⚙️ Retrieval Settings")
        
        retrieval_method = st.selectbox(
            "Retrieval Method",
            ["hybrid", "semantic", "keyword"]
        )
        
        if retrieval_method == "hybrid":
            alpha = st.slider("Semantic Weight (α)", 0.0, 1.0, 0.7)
        else:
            alpha = 0.7
        
        top_k = st.slider("Top K Documents", 1, 10, 5)
    
        st.markdown("---")
        st.header("🧪 Evaluation Settings")
        enable_evaluation = st.checkbox(
            "Enable Evaluation Metrics",
            value=st.session_state.enable_evaluation,
            help="Track answer quality metrics (may increase latency)"
        )
        use_llm_judge = st.checkbox(
            "Use LLM Judge",
            value=st.session_state.use_llm_judge,
            help="Use LLM to evaluate answer quality"
        )
        st.session_state.enable_evaluation = enable_evaluation
        st.session_state.use_llm_judge = use_llm_judge

        st.markdown("---")
        st.header("📊 System Status")
        st.metric("Status", "🟢 Ready")
        st.metric("Documents", len(st.session_state.documents))
        st.metric("Files Uploaded", len(st.session_state.uploaded_files))
    else:
        st.warning("⚠️ Upload files and enter API key to start")

# Main content
if not st.session_state.api_key:
    st.warning("🔑 Please enter your OpenAI API Key in the sidebar to continue")
    st.info("""
    **How to get started:**
    1. Enter your OpenAI API key in the sidebar
    2. Upload one or more documents (TXT, JSON, CSV, Excel, PDF)
    3. Click "Process Files" to index documents
    4. Start asking questions!
    
    **Supported file formats:**
    - **TXT**: Plain text files
    - **JSON**: Array of documents with 'content' field
    - **CSV/Excel**: Must have 'content', 'text', or 'description' column
    - **PDF**: Extracts all text from PDF pages
    """)

elif not st.session_state.rag_engine:
    st.info("📁 Please upload documents in the sidebar to begin")

else:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Q&A", "📚 Documents", "📊 Analytics", "🎯 Evaluation", "🔍 Advanced"])
    
    with tab1:
        st.header("Ask Questions with Multi-Context")
        
        col1, col2 = st.columns([4, 1])
        with col1:
            question = st.text_area(
                "Your Question",
                placeholder="e.g., Analyze the data and calculate the total revenue. Show me documents about X and Y.",
                height=100
            )
        with col2:
            st.write("")
            st.write("")
            enable_calc = st.checkbox("Enable Calculations", value=True)
            multi_context = st.checkbox("Multi-Context Mode", value=True)
        
        col1, col2 = st.columns([1, 5])
        with col1:
            submit = st.button("🚀 Ask", type="primary")
        with col2:
            clear = st.button("🗑️ Clear History")
        
        if clear:
            st.session_state.history = []
            st.experimental_rerun()
        
        # Query processing dengan metrics yang ditingkatkan
        if submit and question:
            with st.spinner("🔍 Searching documents and generating answer..."):
                rag_engine = st.session_state.rag_engine
                rag_engine.top_k = top_k
                rag_engine.hybrid_alpha = alpha
                
                # Set calculation mode in LLM client
                rag_engine.llm_client.enable_calculations = enable_calc
                rag_engine.llm_client.multi_context = multi_context
                
                # Set evaluation mode
                if enable_evaluation:
                    rag_engine.enable_evaluation = True
                    if not rag_engine.evaluator:
                        from src.evaluation_metrics import RAGEvaluator
                        rag_engine.evaluator = RAGEvaluator(
                            llm_client=rag_engine.llm_client if use_llm_judge else None,
                            use_llm_judge=use_llm_judge
                        )
                else:
                    rag_engine.enable_evaluation = False
                
                # Gunakan decorator metrics jika available, jika tidak gunakan function biasa
                if METRICS_AVAILABLE:
                    @monitor_query
                    def execute_query(q):
                        return rag_engine.query(q, retrieval_method)
                else:
                    def execute_query(q):
                        return rag_engine.query(q, retrieval_method)
                
                try:
                    result = execute_query(question)

                    try:
                        metrics = result['metrics']
                        evaluation = result.get('evaluation', {})
                        
                        # Prepare metrics untuk Prometheus
                        prometheus_metrics = {
                            'faithfulness': evaluation.get('generation', {}).get('faithfulness', 0.5),
                            'relevance': evaluation.get('generation', {}).get('relevance', 0.5),
                            'hallucination_rate': evaluation.get('end_to_end', {}).get('hallucination_rate', 0.1),
                            'correctness': evaluation.get('end_to_end', {}).get('correctness', 0.5),
                            'overall_score': evaluation.get('overall_score', 0.5),
                            'evaluation_enabled': enable_evaluation
                        }
                        
                        # Tambahkan retrieval metrics jika ada
                        if 'retrieval' in evaluation:
                            prometheus_metrics['retrieval'] = evaluation['retrieval']
                        
                        # Record metrics untuk Prometheus - GUNAKAN PARAMETER YANG KONSISTEN
                        record_rag_query(
                            retrieval_method=retrieval_method,
                            latency_ms=metrics['total_time_ms'],
                            tokens=metrics.get('tokens', 0),
                            num_documents=len(result['retrieved_documents']),
                            answer=result['answer'],
                            metrics=prometheus_metrics,  # Semua metrics termasuk faithfulness, relevance sudah di sini
                            status='success'
                        )
                        
                        # Set system info (hanya sekali) - GUNAKAN METHOD YANG KONSISTEN
                        if not hasattr(st.session_state, 'system_info_set'):
                            manager = get_metrics_manager()
                            manager.set_system_info(
                                version='2.0.0',
                                embedding_model='all-MiniLM-L6-v2',
                                embedding_dimension=384,  # Dimension for all-MiniLM-L6-v2
                                llm_model='gpt-4',
                                document_count=len(st.session_state.documents)
                            )
                            st.session_state.system_info_set = True
                            
                    except Exception as e:
                        logger.error(f"Metrics recording failed: {e}")
                    
                    # Simpan result ke session state
                    st.session_state.history.append({
                        'timestamp': datetime.now(),
                        'question': question,
                        'result': result,
                        'settings': {
                            'retrieval_method': retrieval_method,
                            'top_k': top_k,
                            'alpha': alpha,
                            'calculations': enable_calc,
                            'multi_context': multi_context,
                            'evaluation_enabled': enable_evaluation
                        }
                    })
                                      
                    st.success("✅ Answer generated!")
                    
                except Exception as e:
                    st.error(f"❌ Error generating answer: {str(e)}")
                    logger.error(f"Query execution failed: {e}")
        
        # Display chat history
        if st.session_state.history:
            st.markdown("---")
            st.subheader("💬 Conversation History")
            
            for idx, item in enumerate(reversed(st.session_state.history)):
                with st.container():
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.markdown(f"**Q #{len(st.session_state.history)-idx}:** {item['question']}")
                    with col2:
                        st.caption(item['timestamp'].strftime("%H:%M:%S"))
                    
                    st.markdown(f"**A:** {item['result']['answer']}")
                    
                    # Metrics
                    metrics = item['result']['metrics']
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("Retrieval", f"{metrics['retrieval_time_ms']:.0f}ms")
                    col2.metric("Generation", f"{metrics['generation_time_ms']:.0f}ms")
                    col3.metric("Total", f"{metrics['total_time_ms']:.0f}ms")
                    col4.metric("Docs", metrics['num_retrieved'])
                    col5.metric("Tokens", metrics.get('tokens', 0))
                    
                    # Retrieved documents
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**📄 Retrieved Context**")
                        for i, doc in enumerate(item['result']['retrieved_documents']):
                            with st.expander(f"Document {i+1}: {doc['title']} (score: {doc['score']:.3f})"):
                                st.markdown(f"**Source:** {doc['metadata'].get('source', 'Unknown')}")
                                st.markdown(f"**Content:** {doc['content'][:500]}...")

                    with col2:
                        if 'evaluation' in item['result']:
                            st.markdown("**📊 Evaluation Metrics**")
                            eval_data = item['result']['evaluation']
                            
                            # Generator metrics
                            if 'generation' in eval_data:
                                st.subheader("🤖 Generation Quality")
                                gen_metrics = eval_data['generation']
                                
                                if 'faithfulness' in gen_metrics:
                                    st.metric("Faithfulness", f"{gen_metrics['faithfulness']:.2%}")
                                if 'relevance' in gen_metrics:
                                    st.metric("Relevance", f"{gen_metrics['relevance']:.2%}")
                            
                            # End-to-end metrics
                            if 'end_to_end' in eval_data:
                                st.subheader("🎯 End-to-End Quality")
                                e2e_metrics = eval_data['end_to_end']
                                
                                col1, col2 = st.columns(2)
                                if 'correctness' in e2e_metrics:
                                    col1.metric("Correctness", f"{e2e_metrics['correctness']:.2%}")
                                if 'hallucination_rate' in e2e_metrics:
                                    halluc_rate = e2e_metrics['hallucination_rate']
                                    col2.metric("Hallucination Rate", f"{halluc_rate:.2%}")
                            
                            # Overall score
                            if 'overall_score' in eval_data:
                                st.subheader("⭐ Overall Quality Score")
                                overall = eval_data['overall_score']
                                st.progress(overall)
                                st.metric("Score", f"{overall:.2%}")
                    
                    st.markdown("---")
    
    with tab2:
        st.header("📚 Uploaded Documents")
        
        if st.session_state.documents:
            # Buat DataFrame untuk documents
            docs_data = []
            for doc in st.session_state.documents:
                docs_data.append({
                    'ID': doc['id'],
                    'Title': doc['title'],
                    'Category': doc['category'],
                    'Content Preview': doc['content'][:100] + '...' if len(doc['content']) > 100 else doc['content'],
                    'Source': doc['metadata'].get('source', 'Unknown'),
                    'Upload Date': doc['metadata'].get('upload_date', 'Unknown')[:10]
                })
            
            docs_df = pd.DataFrame(docs_data)
            
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                category_filter = st.multiselect(
                    "Filter by Category",
                    options=docs_df['Category'].unique(),
                    default=docs_df['Category'].unique()
                )
            with col2:
                source_filter = st.multiselect(
                    "Filter by Source",
                    options=docs_df['Source'].unique(),
                    default=docs_df['Source'].unique()
                )
            with col3:
                search_term = st.text_input("Search in content", "")
            
            # Apply filters
            filtered_docs = docs_df[
                (docs_df['Category'].isin(category_filter)) &
                (docs_df['Source'].isin(source_filter)) &
                (docs_df['Content Preview'].str.contains(search_term, case=False, na=False) | 
                 docs_df['Title'].str.contains(search_term, case=False, na=False) |
                 (search_term == ""))
            ]
            
            st.write(f"📄 Showing {len(filtered_docs)} of {len(st.session_state.documents)} documents")
            
            # Display documents
            for _, doc_row in filtered_docs.iterrows():
                # Find the full document from session state
                full_doc = next((doc for doc in st.session_state.documents if doc['id'] == doc_row['ID']), None)
                
                if full_doc:
                    with st.expander(f"📄 {doc_row['Title']} - {doc_row['Category']}"):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**ID:** `{full_doc['id']}`")
                            st.markdown(f"**Category:** {full_doc['category']}")
                            st.markdown(f"**Source:** {full_doc['metadata'].get('source', 'Unknown')}")
                        with col2:
                            st.markdown(f"**Uploaded:** {full_doc['metadata'].get('upload_date', 'Unknown')[:10]}")
                            if 'pages' in full_doc['metadata']:
                                st.markdown(f"**Pages:** {full_doc['metadata']['pages']}")
                        
                        st.markdown("---")
                        st.markdown("**Content:**")
                        st.text_area(
                            "Document Content",
                            full_doc['content'],
                            height=200,
                            key=f"content_{full_doc['id']}",
                            label_visibility="collapsed"
                        )
        else:
            st.info("No documents uploaded yet")
    
    with tab3:
        st.header("📊 Analytics Dashboard")
        
        if st.session_state.history:
            # Prepare data for analytics
            history_data = []
            for item in st.session_state.history:
                history_data.append({
                    'timestamp': item['timestamp'],
                    'retrieval_method': item['result']['metrics']['retrieval_method'],
                    'retrieval_time_ms': item['result']['metrics']['retrieval_time_ms'],
                    'generation_time_ms': item['result']['metrics']['generation_time_ms'],
                    'total_time_ms': item['result']['metrics']['total_time_ms'],
                    'num_docs': item['result']['metrics']['num_retrieved'],
                    'tokens': item['result']['metrics'].get('tokens', 0)
                })
            
            history_df = pd.DataFrame(history_data)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Queries", len(history_df))
            col2.metric("Avg Latency", f"{history_df['total_time_ms'].mean():.0f}ms")
            col3.metric("Avg Tokens", f"{history_df['tokens'].mean():.0f}")
            col4.metric("Total Tokens", f"{history_df['tokens'].sum():.0f}")
            
            st.markdown("---")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Latency Breakdown")
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name='Retrieval',
                    x=history_df.index,
                    y=history_df['retrieval_time_ms'],
                    marker_color='lightblue'
                ))
                fig.add_trace(go.Bar(
                    name='Generation',
                    x=history_df.index,
                    y=history_df['generation_time_ms'],
                    marker_color='lightgreen'
                ))
                fig.update_layout(
                    barmode='stack', 
                    height=350,
                    title="Latency by Query",
                    xaxis_title="Query Index",
                    yaxis_title="Time (ms)"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Token Usage Over Time")
                if len(history_df) > 1:
                    fig = px.line(history_df, x='timestamp', y='tokens', 
                                 title='Tokens per Query Over Time')
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Need more queries to show time series")
            
            # Retrieval method analysis
            st.subheader("Performance by Retrieval Method")
            method_stats = history_df.groupby('retrieval_method').agg({
                'total_time_ms': ['mean', 'std'],
                'tokens': 'mean',
                'num_docs': 'mean'
            }).round(2)
            
            st.dataframe(method_stats, use_container_width=True)
            
        else:
            st.info("No queries yet. Start asking questions to see analytics!")
    
    with tab4:
        st.header("🎯 Evaluation Dashboard")
        
        # Filter history dengan evaluations
        evaluated_queries = [
            item for item in st.session_state.history
            if 'evaluation' in item['result']
        ]
        
        if not evaluated_queries:
            st.info("No evaluated queries yet. Enable evaluation in settings to track quality metrics.")
            st.markdown("""
            **Evaluation Metrics Explained:**
            
            **Retriever Metrics:**
            - **Recall@K**: % of relevant docs retrieved in top-K
            - **Precision@K**: % of retrieved docs that are relevant
            - **MRR**: Mean Reciprocal Rank of first relevant doc
            - **nDCG@K**: Normalized Discounted Cumulative Gain (ranking quality)
            
            **Generator Metrics:**
            - **Faithfulness**: How well answer is supported by context (0-1)
            - **Relevance**: How well answer addresses the question (0-1)
            
            **End-to-End Metrics:**
            - **Correctness**: Accuracy compared to ground truth (0-1)
            - **Hallucination Rate**: % of claims not supported by context
            
            Enable "Evaluation Metrics" in the sidebar to start tracking these metrics.
            """)
        else:
            # Aggregate metrics
            st.subheader("📈 Aggregated Quality Metrics")
            
            all_faithfulness = []
            all_relevance = []
            all_hallucination = []
            all_correctness = []
            all_overall = []
            
            for item in evaluated_queries:
                eval_data = item['result']['evaluation']
                
                if 'generation' in eval_data:
                    if 'faithfulness' in eval_data['generation']:
                        all_faithfulness.append(eval_data['generation']['faithfulness'])
                    if 'relevance' in eval_data['generation']:
                        all_relevance.append(eval_data['generation']['relevance'])
                
                if 'end_to_end' in eval_data:
                    if 'hallucination_rate' in eval_data['end_to_end']:
                        all_hallucination.append(eval_data['end_to_end']['hallucination_rate'])
                    if 'correctness' in eval_data['end_to_end']:
                        all_correctness.append(eval_data['end_to_end']['correctness'])
                
                if 'overall_score' in eval_data:
                    all_overall.append(eval_data['overall_score'])
            
            # Display aggregate metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            if all_faithfulness:
                avg_faith = sum(all_faithfulness) / len(all_faithfulness)
                col1.metric("Avg Faithfulness", f"{avg_faith:.2%}")
            
            if all_relevance:
                avg_rel = sum(all_relevance) / len(all_relevance)
                col2.metric("Avg Relevance", f"{avg_rel:.2%}")
            
            if all_hallucination:
                avg_hall = sum(all_hallucination) / len(all_hallucination)
                col3.metric("Avg Hallucination", f"{avg_hall:.2%}")
            
            if all_correctness:
                avg_corr = sum(all_correctness) / len(all_correctness)
                col4.metric("Avg Correctness", f"{avg_corr:.2%}")
            
            if all_overall:
                avg_overall = sum(all_overall) / len(all_overall)
                col5.metric("Overall Quality", f"{avg_overall:.2%}")
            
            # Individual query evaluations
            st.markdown("---")
            st.subheader("📋 Detailed Evaluations")
            
            for item in evaluated_queries:
                with st.expander(f"Query: {item['question'][:50]}..."):
                    eval_data = item['result']['evaluation']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'generation' in eval_data:
                            st.markdown("**Generation Quality:**")
                            gen_metrics = eval_data['generation']
                            if 'faithfulness' in gen_metrics:
                                st.metric("Faithfulness", f"{gen_metrics['faithfulness']:.2%}")
                            if 'relevance' in gen_metrics:
                                st.metric("Relevance", f"{gen_metrics['relevance']:.2%}")
                    
                    with col2:
                        if 'end_to_end' in eval_data:
                            st.markdown("**End-to-End Quality:**")
                            e2e_metrics = eval_data['end_to_end']
                            if 'correctness' in e2e_metrics:
                                st.metric("Correctness", f"{e2e_metrics['correctness']:.2%}")
                            if 'hallucination_rate' in e2e_metrics:
                                st.metric("Hallucination Rate", f"{e2e_metrics['hallucination_rate']:.2%}")
    
    with tab5:
        st.header("🔍 Advanced Features")
        
        st.subheader("🧮 Calculation Examples")
        st.markdown("""
        The system can handle calculations and data analysis:
        
        **Example queries:**
        - "Calculate the sum of all revenue values in the documents"
        - "What is the average temperature mentioned across all files?"
        - "Compare values from document A and document B"
        - "Analyze the trend and calculate growth rate"
        """)
        
        st.subheader("🔄 Multi-Context Mode")
        st.markdown("""
        When enabled, the system:
        - Retrieves documents from multiple sources
        - Synthesizes information across documents
        - Provides comprehensive answers with cross-references
        - Better for complex queries spanning multiple topics
        """)
        
        st.subheader("📊 Export Options")
        if st.session_state.history:
            # Export query history
            export_data = []
            for item in st.session_state.history:
                export_data.append({
                    'timestamp': item['timestamp'].isoformat(),
                    'question': item['question'],
                    'answer': item['result']['answer'],
                    'retrieval_method': item['result']['metrics']['retrieval_method'],
                    'latency_ms': item['result']['metrics']['total_time_ms'],
                    'tokens': item['result']['metrics'].get('tokens', 0)
                })
            
            export_df = pd.DataFrame(export_data)
            
            col1, col2 = st.columns(2)
            with col1:
                csv = export_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download History (CSV)",
                    csv,
                    "query_history.csv",
                    "text/csv"
                )
            with col2:
                json_str = json.dumps(export_data, indent=2)
                st.download_button(
                    "📥 Download History (JSON)",
                    json_str,
                    "query_history.json",
                    "application/json"
                )
        
        st.subheader("🔧 System Information")
        st.json({
            "documents_loaded": len(st.session_state.documents),
            "queries_in_history": len(st.session_state.history),
            "evaluated_queries": len([h for h in st.session_state.history if 'evaluation' in h['result']]),
            "rag_engine_initialized": st.session_state.rag_engine is not None
        })

st.markdown("---")
st.caption("🤖 Hybrid RAG System v2.0 - Multi-File Support with GPT-4")