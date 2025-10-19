# RAG System Evaluation Guide

Complete guide for evaluating Retrieval-Augmented Generation systems.

## Table of Contents

1. [Overview](#overview)
2. [Metrics Explained](#metrics-explained)
3. [Evaluation Methods](#evaluation-methods)
4. [Running Evaluations](#running-evaluations)
5. [Interpreting Results](#interpreting-results)
6. [Best Practices](#best-practices)

---

## Overview

RAG evaluation covers three components:

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  RETRIEVER  │────▶│  GENERATOR   │────▶│ END-TO-END  │
│             │     │              │     │             │
│ Recall@K    │     │ Faithfulness │     │ Correctness │
│ Precision@K │     │ Relevance    │     │ Hallucination│
│ MRR, nDCG   │     │              │     │             │
└─────────────┘     └──────────────┘     └─────────────┘
```

**Why Evaluate?**
- Detect quality degradation
- Compare different approaches
- Optimize hyperparameters
- Build confidence in production
- Track improvements over time

---

## Metrics Explained

### Retriever Metrics

#### Recall@K

**Definition:** What proportion of relevant documents were retrieved in top-K?

**Formula:**
```
Recall@K = (# relevant docs in top-K) / (# total relevant docs)
```

**Example:**
- Query: "What is machine learning?"
- Relevant docs: [doc1, doc2, doc3]
- Retrieved top-5: [doc1, doc4, doc2, doc5, doc6]
- Relevant in top-5: [doc1, doc2] = 2
- Recall@5 = 2/3 = 0.67 (67%)

**Interpretation:**
- **0.0**: No relevant docs found
- **0.5**: Found half of relevant docs
- **1.0**: Found all relevant docs

**Good Score:** > 0.80 at K=5

**Use Case:** Measures completeness of retrieval

---

#### Precision@K

**Definition:** What proportion of retrieved documents are relevant?

**Formula:**
```
Precision@K = (# relevant docs in top-K) / K
```

**Example:**
- Retrieved top-5: [doc1, doc4, doc2, doc5, doc6]
- Relevant: [doc1, doc2]
- Precision@5 = 2/5 = 0.40 (40%)

**Interpretation:**
- **0.0**: No retrieved docs are relevant
- **0.5**: Half of retrieved docs are relevant
- **1.0**: All retrieved docs are relevant

**Good Score:** > 0.60 at K=5

**Use Case:** Measures retrieval accuracy

---

#### Mean Reciprocal Rank (MRR)

**Definition:** Average of reciprocal ranks of first relevant document

**Formula:**
```
MRR = 1 / (rank of first relevant doc)
```

**Example:**
- Retrieved: [doc4, doc1, doc5, doc2, doc6]
- First relevant: doc1 at rank 2
- MRR = 1/2 = 0.5

**Interpretation:**
- **1.0**: Relevant doc at rank 1 (perfect)
- **0.5**: Relevant doc at rank 2
- **0.1**: Relevant doc at rank 10

**Good Score:** > 0.70

**Use Case:** Measures how quickly users find relevant info

---

#### nDCG@K (Normalized Discounted Cumulative Gain)

**Definition:** Ranking quality with graded relevance scores

**Formula:**
```
DCG@K = Σ (2^rel_i - 1) / log2(i + 1)
nDCG@K = DCG@K / IDCG@K
```

**Relevance Grades:**
- 0: Not relevant
- 1: Slightly relevant
- 2: Relevant
- 3: Highly relevant

**Example:**
- Retrieved: [doc1, doc2, doc3]
- Relevance: [3, 1, 2]
- DCG = (2³-1)/log2(2) + (2¹-1)/log2(3) + (2²-1)/log2(4)
- DCG = 7.0 + 0.63 + 1.5 = 9.13
- IDCG = 9.5 (perfect ranking: [3, 2, 1])
- nDCG = 9.13 / 9.5 = 0.96

**Interpretation:**
- **1.0**: Perfect ranking
- **0.8-1.0**: Excellent
- **0.6-0.8**: Good
- **<0.6**: Needs improvement

**Good Score:** > 0.75 at K=5

**Use Case:** Best metric for ranking quality with graded relevance

---

### Generator Metrics

#### Faithfulness

**Definition:** How well is the answer supported by retrieved context?

**Calculation:**
```
Faithfulness = (# supported claims) / (# total claims)
```

**Example:**

**Context:**
- "Q1 revenue was $450,000"
- "Q2 revenue was $520,000"

**Answer:** 
- "Q1 revenue was $450,000" ✓ Supported
- "Q2 revenue was $520,000" ✓ Supported  
- "Q3 is projected to be $600,000" ✗ Not in context

**Faithfulness = 2/3 = 0.67**

**Interpretation:**
- **1.0**: All claims supported by context
- **0.8-1.0**: Highly faithful, minimal hallucination
- **0.6-0.8**: Some unsupported claims
- **<0.6**: High hallucination risk

**Good Score:** > 0.85

**Use Case:** Detect hallucinations and ensure factual grounding

---

#### Relevance

**Definition:** How well does the answer address the question?

**Calculation:**
- Lexical overlap between question and answer
- Question type matching (yes/no, factual, how-to)
- Answer completeness

**Example:**

**Question:** "What was Q1 2024 revenue?"

**Answer A:** "Q1 2024 revenue was $450,000" 
- Relevance: 1.0 (directly answers)

**Answer B:** "The company performed well in Q1"
- Relevance: 0.3 (vague, doesn't answer)

**Interpretation:**
- **1.0**: Directly and completely answers
- **0.7-0.9**: Addresses question but incomplete
- **0.4-0.7**: Partially relevant
- **<0.4**: Off-topic

**Good Score:** > 0.80

**Use Case:** Ensure answers actually address user questions

---

### End-to-End Metrics

#### Correctness

**Definition:** How accurate is the answer compared to ground truth?

**Calculation:**
- F1-score between answer and ground truth
- Semantic similarity
- Fact matching

**Example:**

**Ground Truth:** "Q1 revenue was $450,000 with 15% growth"

**Answer A:** "Q1 revenue was $450,000 with 15% growth"
- Correctness: 1.0 (perfect match)

**Answer B:** "Q1 revenue was $450,000"
- Correctness: 0.7 (partial - missing growth)

**Answer C:** "Revenue was $520,000"
- Correctness: 0.2 (wrong fact)

**Interpretation:**
- **1.0**: Perfect accuracy
- **0.8-1.0**: Excellent, minor differences
- **0.6-0.8**: Good, some missing info
- **<0.6**: Significant errors

**Good Score:** > 0.75

**Use Case:** Validate factual accuracy against known answers

---

#### Hallucination Rate

**Definition:** Proportion of answer that's not supported by context

**Calculation:**
```
Hallucination Rate = (# unsupported claims) / (# total claims)
```

**Example:**

**Context:** "Product A costs $100. Product B costs $150."

**Answer:** 
- "Product A costs $100" ✓
- "Product B costs $150" ✓
- "Product C costs $200" ✗ Hallucination!

**Hallucination Rate = 1/3 = 0.33 (33%)**

**Interpretation:**
- **0.0**: No hallucinations (perfect)
- **0.0-0.1**: Excellent, minimal issues
- **0.1-0.3**: Acceptable, some fabrication
- **>0.3**: High risk, needs investigation

**Good Score:** < 0.10 (lower is better)

**Use Case:** Critical for production - detect fabricated information

---

## Evaluation Methods

### Method 1: Rule-Based Evaluation

**How it works:**
- Lexical overlap (word matching)
- N-gram comparison
- Heuristic rules

**Pros:**
- ✅ Fast (milliseconds)
- ✅ Free (no API calls)
- ✅ Deterministic
- ✅ Good for development

**Cons:**
- ❌ Less accurate
- ❌ Misses semantic meaning
- ❌ Can't understand nuance

**When to use:**
- Development and debugging
- Quick iterations
- Budget constraints
- High-volume testing

**Example:**
```python
from src.evaluation_metrics import GeneratorMetrics

metrics = GeneratorMetrics()
faithfulness = metrics.faithfulness_score(answer, context_docs)
relevance = metrics.relevance_score(answer, question)
```

---

### Method 2: LLM-as-Judge

**How it works:**
- Uses GPT-4 to evaluate quality
- Provides reasoning
- More semantic understanding

**Pros:**
- ✅ More accurate
- ✅ Understands semantics
- ✅ Provides explanations
- ✅ Better for nuanced evaluation

**Cons:**
- ❌ Slower (2-5s per eval)
- ❌ Expensive (~2x token cost)
- ❌ Non-deterministic
- ❌ Requires API access

**When to use:**
- Final evaluation
- Quality benchmarking
- Research and analysis
- When accuracy is critical

**Example:**
```python
from src.evaluation_metrics import GeneratorMetrics
from src.llm_client import LLMClient

llm = LLMClient(api_key="sk-...")
metrics = GeneratorMetrics(llm_client=llm)

faithfulness, reasoning = metrics.faithfulness_llm_judge(answer, context_docs)
print(f"Score: {faithfulness:.2f}")
print(f"Reasoning: {reasoning}")
```

---

### Method 3: Hybrid Approach

**Best Practice:**
- Use rule-based for development
- Use LLM-judge for final validation
- Sample-based LLM evaluation

**Example:**
```python
# Development: Fast rule-based
evaluator = RAGEvaluator(llm_client=None, use_llm_judge=False)

# Production: Sample 10% with LLM judge
import random
if random.random() < 0.1:
    evaluator = RAGEvaluator(llm_client=llm, use_llm_judge=True)
```

---

## Running Evaluations

### Option 1: Streamlit UI

**Enable in sidebar:**
1. Check "Enable Evaluation Metrics"
2. Optionally check "Use LLM-as-Judge"
3. Ask questions
4. View metrics in expandable sections

**View dashboard:**
- Go to "Evaluation" tab
- See aggregated metrics
- Export reports

---

### Option 2: Batch Evaluation Script

**Run test set:**
```bash
# Basic evaluation (rule-based)
python run_evaluation.py --api-key sk-...

# With LLM judge
python run_evaluation.py --api-key sk-... --llm-judge

# Custom output
python run_evaluation.py \
  --api-key sk-... \
  --llm-judge \
  --output results.json
```

**Output:**
- JSON: Full detailed results
- CSV: Summary table
- Console: Pretty-printed summary

---

### Option 3: Programmatic

```python
from src.rag_engine import RAGEngine
from src.vector_store import VectorStore
from src.llm_client import LLMClient

# Initialize
vector_store = VectorStore()
vector_store.load_documents('data/documents.json')

llm_client = LLMClient(api_key="sk-...", model="gpt-4")

rag_engine = RAGEngine(
    vector_store,
    llm_client,
    enable_evaluation=True,
    use_llm_judge=True  # Optional
)

# Single query evaluation
result = rag_engine.query_with_evaluation(
    question="What is machine learning?",
    ground_truth="Machine learning is...",
    relevant_doc_ids={'doc1', 'doc2'}
)

print(result['evaluation'])

# Batch evaluation
test_cases = [
    {
        'question': "...",
        'ground_truth': "...",
        'relevant_docs': ['doc1']
    },
    # ... more cases
]

batch_results = rag_engine.batch_evaluate(test_cases)
print(batch_results['aggregated_metrics'])
```

---

## Interpreting Results

### Good Baseline Scores

| Metric | Excellent | Good | Needs Work |
|--------|-----------|------|------------|
| Recall@5 | >0.90 | 0.70-0.90 | <0.70 |
| Precision@5 | >0.80 | 0.60-0.80 | <0.60 |
| MRR | >0.80 | 0.60-0.80 | <0.60 |
| nDCG@5 | >0.85 | 0.70-0.85 | <0.70 |
| Faithfulness | >0.90 | 0.75-0.90 | <0.75 |
| Relevance | >0.85 | 0.70-0.85 | <0.70 |
| Correctness | >0.80 | 0.65-0.80 | <0.65 |
| Hallucination | <0.05 | 0.05-0.15 | >0.15 |

### Diagnostic Patterns

**Low Recall + High Precision:**
- Retriever is too conservative
- Increase top-K
- Lower similarity threshold
- Try hybrid search with more keyword weight

**High Recall + Low Precision:**
- Retriever is too broad
- Decrease top-K
- Increase similarity threshold
- Use more semantic search

**Low Faithfulness + High Relevance:**
- LLM adds extra information
- Strengthen system prompt
- Emphasize "only use context"
- Check for over-creative temperature

**High Faithfulness + Low Relevance:**
- LLM too conservative
- Improve prompt engineering
- Check if question is answerable
- Retrieve more diverse docs

**High Hallucination Rate:**
- Critical issue!
- Check retrieved docs quality
- Strengthen faithfulness constraints
- Lower LLM temperature
- Add fact-checking step

---

## Best Practices

### 1. Create Quality Test Sets

**Good test case:**
```json
{
  "question": "What was Q1 revenue?",
  "ground_truth": "Q1 revenue was $450,000",
  "relevant_docs": ["sales_q1"],
  "relevance_scores": {"sales_q1": 3.0},
  "category": "factual"
}
```

**Tips:**
- Cover different question types
- Include edge cases
- Mix easy and hard questions
- Update regularly

### 2. Track Metrics Over Time

```python
# Save metrics after each change
import json
from datetime import datetime

metrics_history = {
    'timestamp': datetime.now().isoformat(),
    'version': 'v1.2',
    'metrics': aggregated_metrics
}

with open('metrics_history.jsonl', 'a') as f:
    f.write(json.dumps(metrics_history) + '\n')
```

### 3. A/B Test Changes

```python
# Compare two retrieval methods
results_hybrid = rag_engine.batch_evaluate(test_cases)
rag_engine.hybrid_alpha = 0.5
results_balanced = rag_engine.batch_evaluate(test_cases)

# Compare
print("Hybrid (0.7):", results_hybrid['aggregated_metrics']['overall_quality_score'])
print("Balanced (0.5):", results_balanced['aggregated_metrics']['overall_quality_score'])
```

### 4. Set Up Alerts

```yaml
# prometheus_alerts.yml
groups:
  - name: rag_quality
    rules:
      - alert: HighHallucinationRate
        expr: rag_e2e_hallucination_rate > 0.15
        for: 5m
        annotations:
          summary: "Hallucination rate too high"
          
      - alert: LowFaithfulness
        expr: rag_generation_faithfulness < 0.75
        for: 10m
        annotations:
          summary: "Faithfulness dropped below threshold"
```

### 5. Sample-Based Monitoring

For production, evaluate sample of queries:

```python
import random

def should_evaluate():
    return random.random() < 0.05  # 5% sampling

if should_evaluate():
    # Full evaluation with LLM judge
    result = rag_engine.query_with_evaluation(...)
else:
    # Fast query without evaluation
    result = rag_engine.query(...)
```

### 6. Human-in-the-Loop

```python
# Flag low-confidence for human review
if result['evaluation']['overall_score'] < 0.7:
    flag_for_human_review(result)
```

### 7. Cost Optimization

**LLM Judge Cost:**
- Evaluation prompt: ~300 tokens
- Evaluation response: ~150 tokens
- Cost per evaluation: ~$0.015 (GPT-4)

**Budget for 1000 evals:**
- Rule-based: $0 (instant)
- LLM judge: ~$15 (5-10 minutes)

**Recommendation:**
- Development: Rule-based
- CI/CD: Rule-based on PR
- Weekly: Full LLM judge evaluation
- Production: 5% sampling with LLM judge

---

## Troubleshooting

**Q: Metrics seem inconsistent**
- Use LLM judge for ground truth
- Check test set quality
- Ensure deterministic evaluation settings

**Q: All scores are low**
- Check if test set is too hard
- Verify documents contain answers
- Review retrieval parameters

**Q: High latency with evaluation**
- Use rule-based for speed
- Sample evaluation (don't evaluate all)
- Run batch evaluation offline

**Q: Hallucination rate varies widely**
- Normal - some questions harder
- Track by category
- Set thresholds per question type

---

## Resources

- [Test Set Template](data/evaluation_test_set.json)
- [Evaluation Runner](run_evaluation.py)
- [Metrics Implementation](src/evaluation_metrics.py)
- [Prometheus Metrics](monitoring/metrics.py)

---

**Last Updated:** October 2024  
**Version:** 2.0.0