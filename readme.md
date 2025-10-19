## Monitoring Best Practices

1. **Set Alerts**: Configure Prometheus alerting rules
2. **Track SLOs**: Monitor P95/P99 latencies
3. **Cost Tracking**: Monitor token usage for LLM costs
4. **Error Rates**: Track failed queries
5. **A/B Testing**: Compare retrieval methods via metrics
6. **Quality Metrics**: Track faithfulness, relevance, hallucination rate

## Evaluation Metrics

### Retriever Metrics

**Recall@K** - Proportion of relevant docs retrieved
```python
recall@5 = relevant_docs_in_top5 / total_relevant_docs
```

**Precision@K** - Proportion of retrieved docs that are relevant
```python
precision@5 = relevant_docs_in_top5 / 5
```

**MRR (Mean Reciprocal Rank)** - Rank of first relevant doc
```python
mrr = 1 / rank_of_first_relevant_doc
```

**nDCG@K** - Normalized Discounted Cumulative Gain
- Measures ranking quality with graded relevance
- Score 0-1, higher is better

### Generator Metrics

**Faithfulness** - How well answer is supported by context
- 0 = Not supported by context
- 1 = Fully supported by context
- Measures hallucination risk

**Relevance** - How well answer addresses question
- 0 = Off-topic
- 1 = Directly addresses question
- Checks if answer is on-point

### End-to-End Metrics

**Correctness** - Accuracy vs ground truth
- Requires reference answer
- F1-style comparison
- 0 = Incorrect, 1 = Fully correct

**Hallucination Rate** - % of unsupported claims
- 0 = No hallucinations
- 1 = All claims unsupported
- Critical for factual accuracy

### Running Evaluation

**Via Streamlit UI:**
1. Enable "Evaluation Metrics" in sidebar
2. Optionally enable "LLM-as-Judge" for better accuracy
3. Ask questions - metrics appear automatically

**Batch Evaluation:**
```bash
# Basic evaluation
python run_evaluation.py --api-key sk-...

# With LLM-as-judge (more accurate)
python run_evaluation.py --api-key sk-... --llm-judge

# Custom output
python run_evaluation.py --api-key sk-... --output my_eval.json
```

**Test Set Format:**
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

### Evaluation Modes

**Rule-Based (Default)**
- ✅ Fast and free
- ✅ Deterministic
- ❌ Less accurate
- Uses lexical overlap and heuristics

**LLM-as-Judge**
- ✅ More accurate
- ✅ Better understands semantics
- ❌ Slower (additional API calls)
- ❌ More expensive (~2x tokens)
- Uses GPT-4 to evaluate quality

**Trade-offs:**
```
Rule-Based:    Fast, cheap, good for development
LLM-Judge:     Accurate, costly, good for final eval
```# Hybrid RAG LLMOps System - Multi-File Edition

Production-ready Retrieval-Augmented Generation system with **multi-file upload**, **OpenAI GPT-4 integration**, **calculation support**, and **multi-context analysis**.

## Features

✨ **Multi-File Upload**: TXT, JSON, CSV, Excel, PDF support  
🧮 **Calculations**: Automatic math and data analysis  
🔗 **Multi-Context**: Cross-document information synthesis  
🤖 **GPT-4 Integration**: Real OpenAI API with streaming  
📊 **Advanced Analytics**: Token tracking, cost estimation, performance metrics  
🔐 **API Key Management**: Secure key input via UI  
📈 **Monitoring**: Prometheus + Grafana dashboards  
🐳 **Containerized**: Docker Compose for easy deployment

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Streamlit  │────▶│  RAG Engine  │────▶│ Vector Store│
│   Frontend  │     │              │     │   (FAISS)   │
└─────────────┘     └──────────────┘     └─────────────┘
       │                    │                     │
       │                    ▼                     │
       │            ┌──────────────┐              │
       │            │  LLM Client  │              │
       │            └──────────────┘              │
       │                                          │
       ▼                                          ▼
┌─────────────┐                          ┌─────────────┐
│ Prometheus  │◀─────────────────────────│   Metrics   │
│             │                          │  Collector  │
└─────────────┘                          └─────────────┘
       │
       ▼
┌─────────────┐
│   Grafana   │
└─────────────┘
```

## Quick Start

### 1. Clone and Setup

```bash
git clone <repo>
cd hybrid-rag-llmops
pip install -r requirements.txt
```

### 2. Get OpenAI API Key

1. Go to https://platform.openai.com/api-keys
2. Create new secret key
3. Copy the key (starts with `sk-`)

### 3. Run Application

**Option A: Local Development**
```bash
# Terminal 1 - Metrics Server
python metrics_server.py

# Terminal 2 - Streamlit
streamlit run frontend/app.py
```

**Option B: Docker Compose**
```bash
docker-compose up --build
```

### 4. Access & Configure

1. Open **http://localhost:8501**
2. Enter your OpenAI API Key in sidebar
3. Upload documents (TXT, JSON, CSV, Excel, PDF)
4. Click "Process Files"
5. Start asking questions!

**Additional Services:**
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Metrics API**: http://localhost:8000/metrics

## Project Structure

```
hybrid-rag-llmops/
├── data/
│   └── documents.json          # Dummy dataset (8 docs)
├── src/
│   ├── __init__.py
│   ├── vector_store.py         # FAISS + hybrid search
│   ├── llm_client.py           # LLM wrapper (mock for demo)
│   └── rag_engine.py           # Main RAG orchestration
├── monitoring/
│   ├── metrics.py              # Prometheus metrics
│   ├── prometheus.yml          # Prometheus config
│   └── grafana_dashboard.json  # Grafana dashboard
├── frontend/
│   └── app.py                  # Streamlit UI
├── docker-compose.yml          # Container orchestration
├── Dockerfile
├── requirements.txt
├── metrics_server.py           # FastAPI metrics endpoint
└── README.md
```

## Usage Examples

### Basic Q&A

```
Q: What is the total revenue mentioned in the documents?

A: Based on the uploaded sales reports:
- Q1 Revenue: $450,000
- Q2 Revenue: $520,000
Total Revenue: $970,000
```

### Multi-Context Analysis

```
Q: Compare Q1 and Q2 performance and calculate growth rate

A: According to Document 1 (Sales Report Q1) and Document 2 (Sales Report Q2):

Q1 2024: $450,000
Q2 2024: $520,000

Growth Calculation:
($520,000 - $450,000) / $450,000 × 100 = 15.56%

Q2 revenue grew by 15.56% compared to Q1.
```

### Calculations Across Documents

```
Q: Calculate total operational costs from all departments

A: Extracting costs from uploaded documents:
- Personnel: $100,000
- Marketing: $45,000
- Infrastructure: $35,000

Total Operational Costs: $180,000
```

### Cross-Document Synthesis

```
Q: Analyze profitability based on revenue and cost data

A: Based on Documents 1-3:

Revenue (Q1 + Q2): $970,000
Operational Costs: $180,000
Gross Profit: $790,000
Profit Margin: 81.4%

The company shows strong profitability with effective cost management.
```

## Supported File Formats

| Format | Example | Requirements |
|--------|---------|--------------|
| **TXT** | `document.txt` | Plain text, UTF-8 encoding |
| **JSON** | `data.json` | Array with `content` field: `[{"content": "..."}]` |
| **CSV** | `data.csv` | Must have `content`, `text`, `description`, or `body` column |
| **Excel** | `data.xlsx` | Same as CSV requirements |
| **PDF** | `report.pdf` | Text-based PDFs (not scanned images) |

**Upload Tips:**
- Multiple files can be uploaded simultaneously
- CSV/Excel: First row must be headers
- JSON: Can include metadata fields (title, category, tags)
- PDFs: Works best with native digital PDFs

**Sample Data Included:**
- `data/sample_data.csv` - Sales, costs, performance metrics
- `data/sample_financial.json` - Financial analysis documents

### Monitoring Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `rag_queries_total` | Counter | Total queries by method/status |
| `rag_retrieval_duration_seconds` | Histogram | Retrieval latency |
| `rag_generation_duration_seconds` | Histogram | LLM generation latency |
| `rag_total_duration_seconds` | Histogram | End-to-end latency |
| `rag_tokens_total` | Counter | Token consumption |
| `rag_documents_retrieved` | Histogram | Docs per query |
| `rag_indexed_documents` | Gauge | Total indexed docs |
| `rag_active_queries` | Gauge | Concurrent queries |

### LLM Client

Currently uses mock responses for demo. To integrate real LLM:

```python
# In src/llm_client.py
from openai import OpenAI

class LLMClient:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def generate(self, prompt, context):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": self._build_prompt(prompt, context)}
            ],
            temperature=self.temperature
        )
        return response.choices[0].message.content
```

## Configuration

### Vector Store

- **Embedding Model**: `all-MiniLM-L6-v2` (384 dims, fast)
- **Index**: FAISS L2 distance
- **Top-K**: 3-5 documents (configurable)

### Retrieval Methods

1. **Semantic**: Pure embedding-based similarity
2. **Keyword**: Term overlap scoring
3. **Hybrid**: Weighted combination (α parameter)

### Monitoring

- **Scrape Interval**: 5s (Prometheus)
- **Retention**: 15 days (default)
- **Grafana Refresh**: 5s

## Performance

With dummy data (8 documents):

| Metric | Value |
|--------|-------|
| Retrieval Latency | ~10-50ms |
| Generation Latency | ~100-200ms (mock) |
| Total Latency | ~150-300ms |
| Throughput | ~3-5 QPS (single instance) |

## Extending the System

### Add Real LLM

Replace mock client with OpenAI/Anthropic/local model:

```python
# Option 1: OpenAI
from openai import OpenAI
client = OpenAI()

# Option 2: Anthropic Claude
from anthropic import Anthropic
client = Anthropic()

# Option 3: Local (Ollama)
from langchain.llms import Ollama
client = Ollama(model="llama2")
```

### Add More Documents

```python
# Load custom documents
documents = [
    {
        "id": "doc_new",
        "title": "Your Title",
        "content": "Your content...",
        "category": "Category",
        "metadata": {...}
    }
]

# Save to data/documents.json
import json
with open('data/documents.json', 'w') as f:
    json.dump(documents, f, indent=2)
```

### Scale Up

```yaml
# docker-compose.yml
services:
  app:
    deploy:
      replicas: 3
    
  # Add load balancer
  nginx:
    image: nginx:latest
    ports:
      - "80:80"
```

## Monitoring Best Practices

1. **Set Alerts**: Configure Prometheus alerting rules
2. **Track SLOs**: Monitor P95/P99 latencies
3. **Cost Tracking**: Monitor token usage for LLM costs
4. **Error Rates**: Track failed queries
5. **A/B Testing**: Compare retrieval methods via metrics

## Troubleshooting

**Issue**: Streamlit can't import modules
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Issue**: Prometheus not scraping metrics
- Check `http://localhost:8000/metrics` is accessible
- Verify prometheus.yml target config

**Issue**: Grafana no data
- Ensure Prometheus datasource is configured
- Check query syntax in Grafana panels

## License

MIT

## Contributing

PRs welcome! Focus areas:
- Better evaluation metrics
- More retrieval strategies
- Production LLM integration
- Advanced caching
- Query optimization