"""
Batch evaluation runner for RAG system
Evaluates system on test set and generates detailed report
"""

import json
import pandas as pd
from src.vector_store import VectorStore
from src.llm_client import LLMClient
from src.rag_engine import RAGEngine
from src.evaluation_metrics import RAGEvaluator
import os
import sys
from datetime import datetime

def load_test_set(filepath: str = 'data/evaluation_test_set.json'):
    """Load evaluation test set"""
    with open(filepath, 'r') as f:
        return json.load(f)

def run_evaluation(
    api_key: str,
    use_llm_judge: bool = False,
    output_file: str = None
):
    """
    Run full evaluation on test set
    
    Args:
        api_key: OpenAI API key
        use_llm_judge: Whether to use LLM-as-judge (more accurate but costly)
        output_file: Path to save results
    """
    print("=" * 70)
    print("RAG SYSTEM EVALUATION")
    print("=" * 70)
    
    # Initialize system
    print("\n[1/5] Initializing RAG system...")
    
    # Load documents
    with open('data/documents.json', 'r') as f:
        json_docs = json.load(f)
    
    csv_docs = []
    df = pd.read_csv('data/sample_data.csv')
    for idx, row in df.iterrows():
        csv_docs.append({
            'id': f'csv_{idx}',
            'title': row['title'],
            'content': row['content'],
            'category': row['category'],
            'metadata': {'tags': row['tags'].split(','), 'source': 'sample_data.csv'}
        })
    
    all_docs = json_docs + csv_docs
    print(f"   ✓ Loaded {len(all_docs)} documents")
    
    # Initialize vector store
    vector_store = VectorStore()
    vector_store.documents = all_docs
    texts = [doc['content'] for doc in all_docs]
    vector_store.embeddings = vector_store.encoder.encode(texts, show_progress_bar=False)
    
    import faiss
    dimension = vector_store.embeddings.shape[1]
    vector_store.index = faiss.IndexFlatL2(dimension)
    vector_store.index.add(vector_store.embeddings.astype('float32'))
    print(f"   ✓ Built vector index")
    
    # Initialize LLM
    llm_client = LLMClient(api_key=api_key, model="gpt-4", temperature=0.7)
    print(f"   ✓ Initialized GPT-4 client")
    
    # Initialize RAG engine with evaluation
    rag_engine = RAGEngine(
        vector_store,
        llm_client,
        top_k=5,
        enable_evaluation=True,
        use_llm_judge=use_llm_judge
    )
    print(f"   ✓ RAG engine ready (LLM judge: {use_llm_judge})")
    
    # Load test set
    print("\n[2/5] Loading test set...")
    test_cases = load_test_set()
    print(f"   ✓ Loaded {len(test_cases)} test cases")
    
    # Run evaluation
    print(f"\n[3/5] Running evaluation on {len(test_cases)} queries...")
    print(f"   (This may take a few minutes...)\n")
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"   [{i}/{len(test_cases)}] {test_case['question'][:60]}...")
        
        result = rag_engine.query(
            question=test_case['question'],
            retrieval_method='hybrid',
            ground_truth=test_case['ground_truth'],
            relevant_doc_ids=set(test_case['relevant_docs']),
            relevance_scores=test_case.get('relevance_scores', {})
        )
        
        result['test_case'] = test_case
        results.append(result)
    
    print("\n[4/5] Aggregating metrics...")
    
    # Aggregate metrics
    aggregated = rag_engine._aggregate_evaluation_metrics(results)
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    
    # Retrieval metrics
    if 'retrieval' in aggregated and aggregated['retrieval']:
        print("\n📍 RETRIEVAL METRICS:")
        ret = aggregated['retrieval']
        for k in [3, 5, 10]:
            if f'recall@{k}' in ret:
                print(f"   Recall@{k:2d}:    {ret[f'recall@{k}']:.2%}")
        for k in [3, 5, 10]:
            if f'precision@{k}' in ret:
                print(f"   Precision@{k:2d}: {ret[f'precision@{k}']:.2%}")
        if 'mrr' in ret:
            print(f"   MRR:          {ret['mrr']:.3f}")
        for k in [3, 5, 10]:
            if f'ndcg@{k}' in ret:
                print(f"   nDCG@{k:2d}:      {ret[f'ndcg@{k}']:.3f}")
    
    # Generation metrics
    if 'generation' in aggregated and aggregated['generation']:
        print("\n🤖 GENERATION METRICS:")
        gen = aggregated['generation']
        if 'faithfulness' in gen:
            print(f"   Faithfulness: {gen['faithfulness']:.2%}")
        if 'relevance' in gen:
            print(f"   Relevance:    {gen['relevance']:.2%}")
    
    # End-to-end metrics
    if 'end_to_end' in aggregated and aggregated['end_to_end']:
        print("\n🎯 END-TO-END METRICS:")
        e2e = aggregated['end_to_end']
        if 'correctness' in e2e:
            print(f"   Correctness:        {e2e['correctness']:.2%}")
        if 'hallucination_rate' in e2e:
            print(f"   Hallucination Rate: {e2e['hallucination_rate']:.2%}")
    
    # Performance metrics
    if 'performance' in aggregated:
        print("\n⚡ PERFORMANCE METRICS:")
        perf = aggregated['performance']
        print(f"   Avg Latency: {perf['avg_latency_ms']:.0f}ms")
        print(f"   P50 Latency: {perf['p50_latency_ms']:.0f}ms")
        print(f"   P95 Latency: {perf['p95_latency_ms']:.0f}ms")
        print(f"   Avg Tokens:  {perf['avg_tokens']:.0f}")
        print(f"   Total Tokens: {perf['total_tokens']}")
    
    # Overall score
    if 'overall_quality_score' in aggregated:
        print("\n⭐ OVERALL QUALITY SCORE:")
        print(f"   {aggregated['overall_quality_score']:.2%}")
    
    print("\n" + "=" * 70)
    
    # Save results
    print("\n[5/5] Saving results...")
    
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"evaluation_results_{timestamp}.json"
    
    output_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'num_test_cases': len(test_cases),
            'use_llm_judge': use_llm_judge,
            'model': 'gpt-4',
            'top_k': 5
        },
        'aggregated_metrics': aggregated,
        'individual_results': [
            {
                'question': r['question'],
                'answer': r['answer'],
                'ground_truth': r['test_case']['ground_truth'],
                'evaluation': r.get('evaluation', {}),
                'metrics': r['metrics']
            }
            for r in results
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"   ✓ Results saved to: {output_file}")
    
    # Also save CSV summary
    csv_file = output_file.replace('.json', '.csv')
    csv_data = []
    for r in results:
        row = {
            'question': r['question'],
            'category': r['test_case']['category'],
            'latency_ms': r['metrics']['total_time_ms'],
            'tokens': r['metrics']['tokens']
        }
        
        if 'evaluation' in r:
            eval_data = r['evaluation']
            if 'generation' in eval_data:
                row['faithfulness'] = eval_data['generation'].get('faithfulness', 0)
                row['relevance'] = eval_data['generation'].get('relevance', 0)
            if 'end_to_end' in eval_data:
                row['correctness'] = eval_data['end_to_end'].get('correctness', 0)
                row['hallucination_rate'] = eval_data['end_to_end'].get('hallucination_rate', 0)
            if 'retrieval' in eval_data:
                row['recall@5'] = eval_data['retrieval'].get('recall@5', 0)
                row['mrr'] = eval_data['retrieval'].get('mrr', 0)
        
        csv_data.append(row)
    
    pd.DataFrame(csv_data).to_csv(csv_file, index=False)
    print(f"   ✓ CSV summary saved to: {csv_file}")
    
    print("\n✅ Evaluation complete!\n")
    
    return aggregated, results

def main():
    """Main evaluation runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run RAG system evaluation')
    parser.add_argument(
        '--api-key',
        type=str,
        help='OpenAI API key (or set OPENAI_API_KEY env var)'
    )
    parser.add_argument(
        '--llm-judge',
        action='store_true',
        help='Use LLM-as-judge for more accurate evaluation (slower, more expensive)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path for results'
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("Error: OpenAI API key required!")
        print("\nProvide via:")
        print("  --api-key YOUR_KEY")
        print("  or set OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    # Run evaluation
    try:
        aggregated, results = run_evaluation(
            api_key=api_key,
            use_llm_judge=args.llm_judge,
            output_file=args.output
        )
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())