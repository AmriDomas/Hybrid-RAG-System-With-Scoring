# Hybrid RAG System With Scoring

Production-ready **Retrieval-Augmented Generation (RAG)** system with multi-file upload, OpenAI GPT-4 integration, advanced scoring mechanisms, and comprehensive monitoring infrastructure. Built for enterprise-grade document analysis and question-answering with real-time performance tracking.

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [Performance Metrics](#performance-metrics)
- [Evaluation System](#evaluation-system)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## ✨ Features

### Core Capabilities

- **Multi-File Upload**: Support for TXT, JSON, CSV, Excel, and PDF formats
- **Hybrid Search**: Combines semantic (embedding-based) and keyword-based retrieval with configurable weighting
- **GPT-4 Integration**: Real OpenAI API with streaming responses and token tracking
- **Multi-Context Analysis**: Cross-document information synthesis and correlation
- **Calculation Engine**: Automatic mathematical operations and statistical analysis on retrieved data
- **Secure API Management**: User-friendly API key input via Streamlit UI

### Monitoring & Operations

- **Prometheus Metrics**: Comprehensive tracking of latency, throughput, token usage, and query success rates
- **Grafana Dashboards**: Real-time visualization of system performance and health metrics
- **Evaluation Framework**: Built-in metrics for retrieval quality (Recall@K, Precision@K, nDCG@K) and response quality (Faithfulness, Relevance, Hallucination Rate)
- **Docker Containerization**: Production-ready deployment via Docker Compose
- **LLM-as-Judge**: Optional advanced evaluation using GPT-4 for semantic quality assessment

## 🏗️ Architecture

The system follows a modular, scalable architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
│              (File Upload, Chat Interface)                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────┴───────────┐
          │                        │
    ┌─────▼────────┐      ┌───────▼──────┐
    │  RAG Engine  │      │ Metrics      │
    │  - Retrieval │      │ Collector    │
    │  - Generation│      └───────┬──────┘
    └─────┬────────┘              │
          │                   ┌───▼─────┐
    ┌─────▼──────────┐       │Prometheus│
    │  Vector Store  │       └───┬─────┘
    │  (FAISS Index) │           │
    └────────────────┘      ┌────▼──────┐
                            │  Grafana  │
                            │ Dashboards│
                            └───────────┘

    ┌──────────────────┐
    │  LLM Client      │
    │  - OpenAI        │
    │  - Anthropic     │
    │  - Local (Ollama)│
    └──────────────────┘
```

**Key Components**:
- **Vector Store**: FAISS-based storage with L2 distance metric for semantic search
- **RAG Engine**: Orchestrates retrieval, ranking, and generation pipelines
- **LLM Client**: Abstraction layer supporting multiple LLM providers
- **Metrics Server**: FastAPI endpoint exposing Prometheus metrics
- **Evaluation Module**: Computes retrieval and generation quality metrics

## 📋 Prerequisites

### System Requirements
- Python 3.9+
- 4GB RAM (8GB+ recommended)
- 2GB disk space for FAISS indices

### Required Credentials
- **OpenAI API Key**: Get from [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- Docker & Docker Compose (optional, for containerized deployment)

### Python Dependencies
- `streamlit>=1.28.0` - UI framework
- `openai>=1.0.0` - LLM API client
- `faiss-cpu>=1.8.0` - Vector similarity search
- `sentence-transformers>=2.2.0` - Embedding generation
- `pandas>=1.5.0` - Data processing
- `prometheus-client>=0.17.0` - Metrics collection
- `fastapi>=0.104.0` - Metrics API server
- `pydantic>=2.0.0` - Data validation
- See `requirements.txt` for complete list

## 🚀 Installation

### Option A: Local Development

```bash
# Clone repository
git clone https://github.com/AmriDomas/Hybrid-RAG-System-With-Scoring.git
cd Hybrid-RAG-System-With-Scoring

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables (optional)
export OPENAI_API_KEY="sk-your-key-here"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Option B: Docker Compose (Recommended for Production)

```bash
# Clone and navigate to directory
git clone https://github.com/AmriDomas/Hybrid-RAG-System-With-Scoring.git
cd Hybrid-RAG-System-With-Scoring

# Build and start services
docker-compose up --build

# Access the application
# Streamlit: http://localhost:8501
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)
```

## 📁 Project Structure

```
hybrid-rag-llmops/
├── data/
│   ├── documents.json              # Embedded dataset (8 documents)
│   ├── sample_data.csv             # Sample financial data
│   └── sample_financial.json        # Sample analysis documents
│
├── src/
│   ├── __init__.py
│   ├── vector_store.py             # FAISS index + hybrid search logic
│   ├── llm_client.py               # LLM abstraction (OpenAI/Anthropic/Local)
│   ├── rag_engine.py               # Core RAG orchestration
│   ├── evaluator.py                # Evaluation metrics computation
│   └── utils.py                    # Helper functions (file parsing, etc.)
│
├── monitoring/
│   ├── metrics.py                  # Prometheus metric definitions
│   ├── prometheus.yml              # Prometheus scrape configuration
│   └── grafana_dashboard.json      # Pre-built Grafana dashboard
│
├── frontend/
│   └── app.py                      # Streamlit UI application
│
├── docker-compose.yml              # Multi-service orchestration
├── Dockerfile                      # Container image definition
├── requirements.txt                # Python dependencies
├── metrics_server.py               # FastAPI metrics endpoint
├── run_evaluation.py               # Batch evaluation script
└── README.md                       # Documentation (this file)
```

## 💡 Usage

### Quick Start (UI)

```bash
# Terminal 1: Start metrics server
python metrics_server.py

# Terminal 2: Launch Streamlit interface
streamlit run frontend/app.py

# Open browser: http://localhost:8501
```

**Workflow in UI**:
1. Enter OpenAI API Key in sidebar
2. Upload documents (TXT, CSV, JSON, Excel, PDF)
3. Click "Process Files" button
4. Ask questions in chat interface
5. View metrics in sidebar (optional evaluation toggle)

### Example Queries & Responses

**Query 1: Financial Data Aggregation**
```
Q: What is the total revenue and costs mentioned across all documents?

A: Based on the uploaded financial documents (data/sample_data.csv and sample_financial.json):

Revenue Summary:
- Q1 2024 Revenue: $450,000
- Q2 2024 Revenue: $520,000
- Total Revenue: $970,000

Operational Costs Breakdown:
- Personnel Costs: $100,000
- Marketing Expenses: $45,000
- Infrastructure Costs: $35,000
- Total Operational Costs: $180,000

Retrieved Documents: sales_report_q1.csv, sales_report_q2.csv, operational_expenses.json
Retrieval Method: Hybrid (Semantic + Keyword, α=0.6)
```

**Query 2: Cross-Document Performance Analysis with Calculations**
```
Q: Compare Q1 and Q2 performance and calculate growth metrics

A: Comparative Analysis (Documents: sales_report_q1.csv, sales_report_q2.csv):

Revenue Performance:
- Q1 2024: $450,000
- Q2 2024: $520,000
- Absolute Growth: $70,000

Growth Rate Calculation:
Revenue Growth % = (520,000 - 450,000) / 450,000 × 100 = 15.56%

Performance Metrics:
- Q2 outperformed Q1 by 15.56%
- Average Quarterly Revenue: $485,000
- Revenue Trend: Positive growth trajectory
- Compound Growth: Strong quarter-over-quarter performance

Key Insight: The 15.56% growth rate indicates healthy business expansion 
with consistent upward momentum in Q2.

Documents Referenced: 2 | Calculation Steps: 4 | Confidence: 0.94
```

**Query 3: Multi-Document Profitability Analysis**
```
Q: Calculate net profitability and analyze cost efficiency across departments

A: Profitability Analysis (Documents: financial_summary.json, operational_expenses.json, 
   departmental_costs.csv):

Financial Overview:
- Total Revenue (Q1 + Q2): $970,000
- Total Operating Costs: $180,000

Profitability Calculation:
- Gross Profit = $970,000 - $180,000 = $790,000
- Profit Margin % = ($790,000 / $970,000) × 100 = 81.44%
- Net Profit per Quarter (Average) = $395,000

Cost Efficiency by Department:
- Personnel: 55.6% of total costs ($100,000)
- Marketing: 25.0% of total costs ($45,000)
- Infrastructure: 19.4% of total costs ($35,000)

Efficiency Insights:
✓ High profit margin (81.44%) indicates strong operational efficiency
✓ Personnel costs are well-managed at 10.3% of revenue
✓ Marketing spend is optimized at 4.6% of revenue
✓ Infrastructure investment sustainable at 3.6% of revenue

Recommendation: Maintain current cost structure; slight optimization 
possible in marketing spend allocation.

Documents Referenced: 3 | Calculations: 7 | Evaluation Score: 0.91
```

**Query 4: Semantic Search with Multi-Context Synthesis**
```
Q: What factors contributed to our sales performance improvement?

A: Sales Performance Contributing Factors (Multi-document synthesis):

Primary Growth Drivers (from sales_report_q2.csv, market_analysis.json):
1. Market Expansion: Entered 2 new regional markets contributing ~$28,000 (40% of growth)
2. Customer Retention: Repeat customers increased from 65% to 78%, adding ~$21,000
3. Product Mix Optimization: New product line contributed ~$21,000

Secondary Factors (from operational_expenses.json):
- Enhanced Marketing Campaign: Cost $45,000 → Generated $70,000 incremental revenue (ROI: 55%)
- Improved Sales Team: Added 3 new account managers → Managed existing + new accounts
- Operational Efficiency: Cost optimization freed up resources for revenue-generating activities

Compound Effect:
All factors combined create the observed 15.56% growth.
Individual contribution: Market Expansion (40%) > Product Mix (30%) > Customer Retention (30%)

Retrieved via Hybrid Search:
- Semantic Match: "sales performance factors" matched with revenue drivers
- Keyword Match: "market", "sales", "growth" found in key documents
- Cross-reference: Linked expenditure (operational_expenses.json) to results (sales_report_q2.csv)

Confidence Score: 0.89 | Hallucination Risk: Low | Relevance: High
```

### Programmatic Usage (Python API)

```python
from src.rag_engine import RAGEngine
from src.vector_store import VectorStore

# Initialize components
vector_store = VectorStore(embedding_model="all-MiniLM-L6-v2")
rag_engine = RAGEngine(vector_store=vector_store, llm_model="gpt-3.5-turbo")

# Load documents
documents = [
    {"id": "doc_1", "title": "Sales Q1", "content": "Q1 revenue: $450,000"},
    {"id": "doc_2", "title": "Sales Q2", "content": "Q2 revenue: $520,000"}
]
vector_store.add_documents(documents)

# Query
query = "What is total revenue?"
context = vector_store.hybrid_search(query, top_k=5, alpha=0.6)
response = rag_engine.generate(query, context)
print(response)
```

### File Format Support

| Format | Extension | Requirements |
|--------|-----------|--------------|
| Text | `.txt` | Plain text, UTF-8 encoding |
| JSON | `.json` | Array with `content` field: `[{"content": "..."}]` |
| CSV | `.csv` | Must have `content`, `text`, `description`, or `body` column |
| Excel | `.xlsx` | Same as CSV (headers required) |
| PDF | `.pdf` | Native digital PDFs (not scanned images) |

**Upload Tips**:
- Multiple files can be uploaded simultaneously
- CSV/Excel must have headers in first row
- JSON can include metadata fields (title, category, tags)
- PDFs work best with native digital text (OCR not supported)

## ⚙️ Configuration

### Retrieval Settings

Edit `frontend/app.py` or pass as parameters:

```python
# Vector Store Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384 dimensions, fast inference
TOP_K = 5                             # Number of docs to retrieve
HYBRID_ALPHA = 0.6                    # Weight for semantic (0.0 = keyword only, 1.0 = semantic only)

# Retrieval Methods:
# - Semantic: Pure embedding similarity (FAISS L2)
# - Keyword: Term overlap + BM25 scoring
# - Hybrid: α * semantic + (1-α) * keyword
```

### LLM Configuration

```python
# Default: OpenAI
LLM_MODEL = "gpt-4"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 1024

# Switch providers (in src/llm_client.py):
# Option 1: OpenAI (default)
# Option 2: Anthropic Claude
# Option 3: Local Ollama
```

### Monitoring Configuration

**Prometheus scrape settings** (`monitoring/prometheus.yml`):
```yaml
global:
  scrape_interval: 5s      # Scrape every 5 seconds
  evaluation_interval: 5s

scrape_configs:
  - job_name: 'rag-metrics'
    static_configs:
      - targets: ['localhost:8000']
```

**Grafana**: Pre-configured dashboard at `http://localhost:3000`
- Default credentials: `admin` / `admin`
- Datasource: Prometheus (http://localhost:9090)

## 📊 Performance Metrics

### Tracked Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `rag_queries_total` | Counter | Total queries by method/status |
| `rag_retrieval_duration_seconds` | Histogram | Retrieval latency (p50, p95, p99) |
| `rag_generation_duration_seconds` | Histogram | LLM generation latency |
| `rag_total_duration_seconds` | Histogram | End-to-end query latency |
| `rag_tokens_total` | Counter | Token consumption (input/output) |
| `rag_documents_retrieved` | Histogram | Number of docs per query |
| `rag_indexed_documents` | Gauge | Total indexed documents |
| `rag_active_queries` | Gauge | Concurrent active queries |

### Baseline Performance (8 documents)

| Metric | Value | Notes |
|--------|-------|-------|
| Retrieval Latency | 10-50ms | Depends on FAISS index size |
| Generation Latency | 100-200ms | Mock responses; real API ~500-2000ms |
| Total Latency | 150-300ms | Mock setup; expect higher in production |
| Throughput | 3-5 QPS | Single instance; scales linearly with Docker replicas |

### Access Metrics

- **Prometheus UI**: http://localhost:9090 (PromQL queries)
- **Metrics API**: http://localhost:8000/metrics (raw Prometheus format)
- **Grafana Dashboard**: http://localhost:3000 (pre-configured visualizations)

## 🎯 Evaluation System

### Retrieval Quality Metrics

**Recall@K** - Proportion of relevant documents retrieved
```
Recall@5 = relevant_docs_in_top5 / total_relevant_docs
Perfect score: 1.0 (all relevant docs in top-5)
```

**Precision@K** - Proportion of retrieved documents that are relevant
```
Precision@5 = relevant_docs_in_top5 / 5
Perfect score: 1.0 (all top-5 are relevant)
```

**MRR (Mean Reciprocal Rank)** - Ranking quality (reciprocal of first relevant doc's position)
```
MRR = 1 / rank_of_first_relevant_doc
Perfect score: 1.0 (first doc is relevant)
```

**nDCG@K (Normalized Discounted Cumulative Gain)** - Ranking quality with graded relevance
```
Score: 0-1 (higher is better)
Accounts for position and relevance scores
```

### Generation Quality Metrics

**Faithfulness** - How well answer is supported by context
- Score 0-1 (1 = fully supported by context)
- Measures hallucination risk
- Lower scores indicate unsupported claims

**Relevance** - How well answer addresses question
- Score 0-1 (1 = directly addresses question)
- Checks if answer is on-topic
- Off-topic responses score lower

**Hallucination Rate** - Percentage of unsupported claims
- Score 0-1 (0 = no hallucinations, 1 = all claims unsupported)
- Critical for factual accuracy
- Computed via semantic entailment

### Running Evaluations

**Via Streamlit UI**:
```
1. Enable "Evaluation Metrics" in sidebar toggle
2. Optionally enable "LLM-as-Judge" for higher accuracy
3. Ask questions - metrics display automatically
```

**Batch Evaluation (CLI)**:
```bash
# Basic evaluation (rule-based, fast and free)
python run_evaluation.py --api-key sk-...

# With LLM-as-Judge (more accurate, slower and costlier)
python run_evaluation.py --api-key sk-... --llm-judge

# Custom output file
python run_evaluation.py --api-key sk-... --output my_eval.json
```

**Test Set Format** (`test_set.json`):
```json
[
  {
    "question": "What was Q1 revenue?",
    "ground_truth": "Q1 revenue was $450,000",
    "relevant_docs": ["csv_0"],
    "relevance_scores": {"csv_0": 3.0},
    "category": "factual"
  }
]
```

### Evaluation Trade-Offs

**Rule-Based Evaluation** (Default)
- ✅ Fast and free
- ✅ Deterministic
- ❌ Less accurate
- Uses lexical overlap and heuristics

**LLM-as-Judge Evaluation**
- ✅ More accurate (semantic understanding)
- ✅ Better at catching subtle hallucinations
- ❌ Slower (~2x more API calls)
- ❌ More expensive (~2x token consumption)
- Uses GPT-4 for quality assessment

## 🔧 Troubleshooting

### Import Errors with Streamlit

**Issue**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
streamlit run frontend/app.py
```

### Prometheus Not Scraping Metrics

**Issue**: Prometheus shows "UP 0" for targets

**Solutions**:
1. Verify metrics endpoint is accessible: `curl http://localhost:8000/metrics`
2. Check `monitoring/prometheus.yml` target configuration
3. Verify `metrics_server.py` is running: `ps aux | grep metrics_server`
4. Check firewall: `telnet localhost 8000`

### Grafana No Data

**Issue**: Grafana dashboard shows empty panels

**Solutions**:
1. Ensure Prometheus datasource configured: Settings → Data Sources
2. Verify datasource URL: `http://prometheus:9090` (Docker) or `http://localhost:9090` (local)
3. Check Grafana panel PromQL queries are valid
4. Wait 60 seconds for Prometheus to scrape metrics (default scrape_interval: 5s)

### FAISS Index Too Large

**Issue**: Memory error when indexing many documents

**Solution**: Switch to disk-based FAISS index
```python
# In src/vector_store.py
import faiss
index = faiss.IndexIDMap(faiss.index_factory(
    384, "IVF100,PQ32", faiss.METRIC_L2
))
# IVF100 = inverted index with 100 clusters (faster, less memory)
# PQ32 = product quantization (32 bytes per vector)
```

### High Latency

**Issue**: Queries taking >1 second

**Causes & Solutions**:
- Too many documents: Reduce `TOP_K` retrieval count
- Slow embedding model: Switch to faster model (e.g., "DistilBERT")
- LLM API rate limit: Add retry logic with exponential backoff
- Network latency: Ensure server close to API endpoints

## 🚀 Production Deployment

### Scaling

```yaml
# docker-compose.yml - Enable auto-scaling
services:
  app:
    deploy:
      replicas: 3  # Run 3 instances
    
  # Add load balancer
  nginx:
    image: nginx:latest
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
```

### Monitoring & Alerting

```bash
# Configure Prometheus alerts
- Set alert thresholds in prometheus.yml
- Monitor P95/P99 latencies for SLOs
- Track token usage for LLM cost estimation
- Alert on high error rates (query failures)
```

### Cost Optimization

```python
# Monitor token usage per query
from src.rag_engine import RAGEngine

# Access token metrics
tokens_used = metrics.rag_tokens_total.labels(type="input").value
estimated_cost = (tokens_used / 1000) * 0.03  # $0.03 per 1K tokens (GPT-3.5)
```

## 🤝 Contributing

Contributions are welcome! Priority areas:
- Advanced retrieval strategies (ColBERT, DPR)
- Better evaluation metrics and automated benchmarks
- Local LLM integration (Llama 2, Mistral)
- Advanced caching strategies (Redis, semantic cache)
- Query optimization and rewriting
- Multi-language support

## 📜 License

MIT License - See LICENSE file for details

## 👤 Contact

**Project Maintainer**: Amri Domas
- GitHub: [@AmriDomas](https://github.com/AmriDomas)
- Repository: [Hybrid-RAG-System-With-Scoring](https://github.com/AmriDomas/Hybrid-RAG-System-With-Scoring)

**For Issues & Questions**:
- GitHub Issues: [Report bugs or request features](https://github.com/AmriDomas/Hybrid-RAG-System-With-Scoring/issues)
- Email: Contact via GitHub profile

**Support & Resources**:
- OpenAI API Documentation: https://platform.openai.com/docs
- FAISS Documentation: https://faiss.ai/
- Streamlit Documentation: https://docs.streamlit.io/
- Prometheus Documentation: https://prometheus.io/docs/

---

**Last Updated**: October 2025  
**Version**: 1.0.0  
**Status**: Production Ready
