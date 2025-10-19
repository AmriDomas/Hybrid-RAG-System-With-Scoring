# Multi-File RAG System - Usage Guide

## Quick Start

### 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables (optional)
cp .env.example .env
# Edit .env with your settings
```

### 2. Run the Application

**Option A: Local Development**
```bash
# Terminal 1 - Metrics Server
python metrics_server.py

# Terminal 2 - Streamlit App
streamlit run frontend/app.py
```

**Option B: Docker Compose**
```bash
docker-compose up --build

# Access:
# Streamlit: http://localhost:8501
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000
```

### 3. Configure API Key

1. Open Streamlit UI at http://localhost:8501
2. Enter your OpenAI API key in the sidebar
3. Click outside the input to save

**Getting an API Key:**
- Go to https://platform.openai.com/api-keys
- Create new secret key
- Copy and paste into Streamlit

---

## Uploading Documents

### Supported File Formats

| Format | Requirements | Example |
|--------|-------------|---------|
| **TXT** | Plain text | `document.txt` |
| **JSON** | Array with 'content' field | `[{"content": "text"}]` |
| **CSV** | Must have 'content' or 'text' column | `data.csv` |
| **Excel** | .xlsx/.xls with 'content' column | `data.xlsx` |
| **PDF** | Text-based PDFs | `report.pdf` |

### Upload Process

1. Click "Browse files" in sidebar
2. Select one or multiple files
3. Click "🚀 Process Files"
4. Wait for indexing to complete

**Tips:**
- Upload multiple files at once for cross-document analysis
- Ensure CSV/Excel files have proper column names
- PDF text extraction works best with native PDFs (not scanned images)

---

## Asking Questions

### Basic Queries

Simple information retrieval:

```
What is the total revenue mentioned in the documents?
```

```
Summarize the key findings from the sales reports
```

### Multi-Context Queries

Questions spanning multiple documents:

```
Compare Q1 and Q2 revenue and calculate the growth rate
```

```
What patterns do you see across all financial documents?
```

```
Cross-reference the cost reports with revenue data
```

### Calculation Queries

Math and data analysis:

```
Calculate the total operational costs across all departments
```

```
What is the average customer satisfaction score?
```

```
Sum all revenue figures and show the breakdown by product
```

```
Calculate the profit margin based on revenue and costs
```

### Advanced Queries

Complex multi-step analysis:

```
Analyze Q1 and Q2 performance, calculate growth rates, 
and project Q3 revenue using the trend
```

```
Extract all numerical metrics from the documents and 
create a summary table with calculations
```

```
Compare employee performance across departments and 
calculate which department has the highest average score
```

---

## Configuration Options

### Retrieval Method

**Semantic Search**
- Uses embeddings for conceptual matching
- Best for: Questions about concepts, themes, similar ideas
- Example: "What documents discuss customer feedback?"

**Keyword Search**
- Exact term matching (BM25-like)
- Best for: Specific terms, names, precise phrases
- Example: "Find documents mentioning Product A"

**Hybrid Search** ⭐ (Recommended)
- Combines both approaches
- Configurable weight (α parameter)
- α = 0.7 means 70% semantic, 30% keyword
- Best for: Most queries

### Top-K Documents

Controls how many documents are retrieved:

- **1-3**: Fast, focused answers (good for simple queries)
- **3-5**: Balanced (recommended default)
- **5-10**: Comprehensive (good for complex analysis)

**Trade-off:**
- More docs = better context but slower + more tokens
- Fewer docs = faster but might miss relevant info

### Calculation Mode

When enabled, GPT-4 will:
- Extract numbers from documents
- Perform mathematical operations
- Show step-by-step work
- Verify results

**Example Output:**
```
Based on the documents:
- Q1 Revenue: $450,000
- Q2 Revenue: $520,000

Calculation:
Growth = (Q2 - Q1) / Q1 × 100
Growth = ($520,000 - $450,000) / $450,000 × 100
Growth = $70,000 / $450,000 × 100
Growth = 15.56%

Result: Q2 revenue grew by 15.56% compared to Q1.
```

### Multi-Context Mode

When enabled, the system:
- Retrieves from multiple documents
- Synthesizes information across sources
- Cites document sources
- Identifies patterns and connections

**Example Response:**
```
According to Document 1 (Sales Report Q1), revenue was $450,000.
Document 2 (Sales Report Q2) shows revenue increased to $520,000.
When cross-referenced with Document 5 (Cost Analysis), 
this represents a net profit improvement of 3%.
```

---

## Example Use Cases

### Financial Analysis

**Scenario:** Analyze quarterly performance

**Files to upload:**
- Q1_sales.csv
- Q2_sales.csv
- costs.json

**Query:**
```
Compare Q1 and Q2 revenue, calculate growth rate, 
and determine profitability after subtracting costs
```

### Market Research

**Scenario:** Competitive analysis

**Files to upload:**
- market_research.txt
- competitor_data.xlsx
- customer_surveys.csv

**Query:**
```
Analyze market positioning, compare our metrics 
to competitors, and identify improvement opportunities
```

### Operations Review

**Scenario:** Multi-department analysis

**Files to upload:**
- hr_metrics.json
- sales_performance.csv
- it_infrastructure.txt

**Query:**
```
Provide an overview of company-wide performance 
across HR, Sales, and IT departments with key metrics
```

---

## Analytics Dashboard

### Metrics Tracked

**Query Metrics:**
- Total queries executed
- Average response latency
- Token consumption
- Query success rate

**Performance Breakdown:**
- Retrieval time vs Generation time
- Latency trends over time
- Method distribution (semantic/keyword/hybrid)

**Cost Tracking:**
- Total tokens used
- Average tokens per query
- Estimated API costs (tokens × $0.03/1K for GPT-4)

### Exporting Data

Export query history in two formats:

**CSV Export:**
- Timestamp, question, answer, metrics
- Good for: Excel analysis, reporting

**JSON Export:**
- Complete structured data
- Good for: Programmatic analysis, archiving

---

## Monitoring with Prometheus & Grafana

### Prometheus Metrics

Access raw metrics: http://localhost:9090

**Key Queries:**

```promql
# Query rate (queries per second)
rate(rag_queries_total[5m])

# Average latency
rate(rag_total_duration_seconds_sum[5m]) 
/ rate(rag_total_duration_seconds_count[5m])

# P95 latency (95th percentile)
histogram_quantile(0.95, 
  rate(rag_total_duration_seconds_bucket[5m]))

# Token usage rate
rate(rag_tokens_total[5m])

# Error rate
rate(rag_queries_total{status="error"}[5m]) 
/ rate(rag_queries_total[5m])
```

### Grafana Dashboard

Access: http://localhost:3000 (admin/admin)

**Pre-configured Panels:**
1. Query Rate - QPS over time
2. Latency Distribution - P50, P95, P99
3. Token Usage - Cumulative and rate
4. Document Retrieval - Avg docs per query
5. System Status - Active queries, indexed docs
6. Error Tracking - Failed queries

---

## Best Practices

### Document Preparation

✅ **Do:**
- Use clear, descriptive titles
- Keep documents focused on single topics
- Include metadata (dates, sources, categories)
- Use consistent formatting

❌ **Avoid:**
- Extremely long documents (split into chunks)
- Binary-only PDFs (scanned images)
- Duplicate content across files
- Missing or unclear column names in CSV/Excel

### Query Formulation

✅ **Effective Queries:**
- "Calculate total revenue from Q1 and Q2 sales reports"
- "Compare employee performance across all departments"
- "Summarize cost savings initiatives and quantify impact"

❌ **Less Effective:**
- "Tell me stuff" (too vague)
- "Give me everything" (too broad)
- One-word queries (need more context)

### Performance Optimization

**For Speed:**
- Reduce top-k to 3
- Use semantic search only
- Disable multi-context for simple queries
- Lower max_tokens if responses are too long

**For Quality:**
- Increase top-k to 5-7
- Use hybrid search
- Enable multi-context mode
- Use calculation mode for numerical data

**For Cost:**
- Monitor token usage in analytics
- Use lower top-k values
- Consider using gpt-3.5-turbo for simple queries
- Cache frequently asked questions

---

## Troubleshooting

### Common Issues

**"Invalid API Key"**
- Verify key is correct (starts with sk-)
- Check OpenAI account has credits
- Ensure no extra spaces when pasting

**"No documents retrieved"**
- Check if files were processed successfully
- Verify query relevance to uploaded content
- Try different retrieval methods
- Increase top-k value

**"Slow response times"**
- Reduce top-k documents
- Check network connection
- Monitor Prometheus for bottlenecks
- Consider using faster model (gpt-3.5-turbo)

**"Out of context length"**
- Too many/large documents retrieved
- Reduce top-k value
- Split large documents into smaller chunks
- Decrease max_tokens setting

### Debug Mode

Enable debug logging:

```python
# In frontend/app.py, add at top:
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check metrics endpoint:
```bash
curl http://localhost:8000/metrics
```

---

## API Cost Estimation

### Token Pricing (OpenAI GPT-4)

| Model | Input | Output |
|-------|-------|--------|
| gpt-4 | $0.03/1K | $0.06/1K |
| gpt-4-turbo | $0.01/1K | $0.03/1K |
| gpt-3.5-turbo | $0.0005/1K | $0.0015/1K |

### Cost Estimation

**Example Query:**
- Top-K: 5 documents
- Avg document: 500 tokens
- Query: 50 tokens
- Response: 300 tokens

**Input tokens:** 5 × 500 + 50 = 2,550 tokens
**Output tokens:** 300 tokens

**Cost (GPT-4):**
- Input: 2,550 × $0.03/1000 = $0.0765
- Output: 300 × $0.06/1000 = $0.018
- **Total:** $0.0945 per query

**Daily estimate (100 queries):**
100 × $0.0945 = **$9.45/day**

**Tips to reduce costs:**
- Use gpt-3.5-turbo for simple queries (10x cheaper)
- Reduce top-k for straightforward questions
- Implement caching for repeated queries
- Monitor token usage in analytics dashboard

---

## Advanced Configuration

### Custom Embedding Models

Change in `src/vector_store.py`:

```python
# Faster but less accurate
vector_store = VectorStore(model_name="all-MiniLM-L6-v2")

# Better accuracy but slower
vector_store = VectorStore(model_name="all-mpnet-base-v2")
```

### Adjust Temperature

In Streamlit sidebar or `src/llm_client.py`:

```python
# More deterministic (0.0-0.3)
llm_client = LLMClient(temperature=0.2)

# Balanced (0.5-0.7)
llm_client = LLMClient(temperature=0.7)

# More creative (0.8-1.0)
llm_client = LLMClient(temperature=0.9)
```

### Hybrid Search Tuning

Alpha parameter (semantic vs keyword weight):

- α = 1.0: Pure semantic search
- α = 0.7: Balanced (recommended)
- α = 0.5: Equal weight
- α = 0.3: Favor keyword matching
- α = 0.0: Pure keyword search

---

## Support & Resources

**Documentation:**
- OpenAI API: https://platform.openai.com/docs
- Sentence Transformers: https://www.sbert.net
- FAISS: https://github.com/facebookresearch/faiss
- Prometheus: https://prometheus.io/docs

**Getting Help:**
- Check error messages in Streamlit
- Review Prometheus metrics for bottlenecks
- Enable debug mode for detailed logs
- Check OpenAI status: https://status.openai.com

**Community:**
- Report issues on GitHub
- Join discussions
- Contribute improvements

---

## Next Steps

1. **Test with sample data:**
   - Upload `data/sample_data.csv`
   - Try example queries from this guide

2. **Upload your own data:**
   - Prepare documents in supported formats
   - Process and index files
   - Experiment with different queries

3. **Optimize performance:**
   - Monitor metrics in Grafana
   - Tune retrieval parameters
   - A/B test different models

4. **Scale up:**
   - Add more documents
   - Increase infrastructure capacity
   - Implement caching layer

5. **Production deployment:**
   - Set up proper monitoring
   - Configure alerting rules
   - Implement rate limiting
   - Add authentication

---

**Version:** 2.0.0  
**Last Updated:** October 2024  
**License:** MIT