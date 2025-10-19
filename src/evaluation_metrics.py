"""
RAG Evaluation Metrics
- Retriever: Recall@K, MRR, nDCG
- Generator: Faithfulness, Relevance
- End-to-End: Correctness, Hallucination Rate
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
import re
from collections import Counter
import math


class RetrieverMetrics:
    """Metrics for evaluating retrieval quality"""
    
    @staticmethod
    def recall_at_k(retrieved_docs: List[str], relevant_docs: Set[str], k: int = None) -> float:
        """
        Recall@K: Proportion of relevant documents retrieved in top-k
        
        Args:
            retrieved_docs: List of retrieved document IDs (ordered by relevance)
            relevant_docs: Set of ground truth relevant document IDs
            k: Cutoff position (if None, uses all retrieved docs)
            
        Returns:
            Recall score [0, 1]
        """
        if not relevant_docs:
            return 0.0
        
        if k is not None:
            retrieved_docs = retrieved_docs[:k]
        
        retrieved_set = set(retrieved_docs)
        relevant_retrieved = retrieved_set.intersection(relevant_docs)
        
        recall = len(relevant_retrieved) / len(relevant_docs)
        return recall
    
    @staticmethod
    def precision_at_k(retrieved_docs: List[str], relevant_docs: Set[str], k: int = None) -> float:
        """
        Precision@K: Proportion of retrieved documents that are relevant
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: Set of ground truth relevant document IDs
            k: Cutoff position
            
        Returns:
            Precision score [0, 1]
        """
        if not retrieved_docs:
            return 0.0
        
        if k is not None:
            retrieved_docs = retrieved_docs[:k]
        
        retrieved_set = set(retrieved_docs)
        relevant_retrieved = retrieved_set.intersection(relevant_docs)
        
        precision = len(relevant_retrieved) / len(retrieved_docs)
        return precision
    
    @staticmethod
    def mean_reciprocal_rank(retrieved_docs: List[str], relevant_docs: Set[str]) -> float:
        """
        MRR: Mean Reciprocal Rank - measures rank of first relevant document
        
        Args:
            retrieved_docs: List of retrieved document IDs (ordered)
            relevant_docs: Set of ground truth relevant document IDs
            
        Returns:
            MRR score [0, 1]
        """
        for rank, doc_id in enumerate(retrieved_docs, start=1):
            if doc_id in relevant_docs:
                return 1.0 / rank
        return 0.0
    
    @staticmethod
    def ndcg_at_k(retrieved_docs: List[str], relevant_docs: Dict[str, float], k: int = None) -> float:
        """
        nDCG@K: Normalized Discounted Cumulative Gain
        Measures ranking quality with graded relevance
        
        Args:
            retrieved_docs: List of retrieved document IDs (ordered)
            relevant_docs: Dict mapping doc_id to relevance score (e.g., 0-3)
            k: Cutoff position
            
        Returns:
            nDCG score [0, 1]
        """
        if k is not None:
            retrieved_docs = retrieved_docs[:k]
        
        # DCG: Discounted Cumulative Gain
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_docs, start=1):
            relevance = relevant_docs.get(doc_id, 0.0)
            dcg += (2 ** relevance - 1) / math.log2(i + 1)
        
        # IDCG: Ideal DCG (perfect ranking)
        ideal_relevances = sorted(relevant_docs.values(), reverse=True)[:len(retrieved_docs)]
        idcg = 0.0
        for i, relevance in enumerate(ideal_relevances, start=1):
            idcg += (2 ** relevance - 1) / math.log2(i + 1)
        
        if idcg == 0:
            return 0.0
        
        ndcg = dcg / idcg
        return ndcg
    
    @staticmethod
    def average_precision(retrieved_docs: List[str], relevant_docs: Set[str]) -> float:
        """
        Average Precision: Average of precision values at each relevant doc position
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: Set of ground truth relevant document IDs
            
        Returns:
            AP score [0, 1]
        """
        if not relevant_docs:
            return 0.0
        
        precisions = []
        num_relevant = 0
        
        for i, doc_id in enumerate(retrieved_docs, start=1):
            if doc_id in relevant_docs:
                num_relevant += 1
                precision_at_i = num_relevant / i
                precisions.append(precision_at_i)
        
        if not precisions:
            return 0.0
        
        return sum(precisions) / len(relevant_docs)


class GeneratorMetrics:
    """Metrics for evaluating generation quality"""
    
    def __init__(self, llm_client=None):
        """
        Initialize generator metrics
        
        Args:
            llm_client: LLM client for LLM-as-judge evaluations
        """
        self.llm_client = llm_client
    
    @staticmethod
    def faithfulness_score(answer: str, context_docs: List[Dict]) -> float:
        """
        Faithfulness: How well the answer is supported by retrieved context
        Uses claim extraction and verification
        
        Args:
            answer: Generated answer
            context_docs: Retrieved documents used for generation
            
        Returns:
            Faithfulness score [0, 1]
        """
        # Extract claims from answer (simplified - split by sentences)
        claims = [s.strip() for s in re.split(r'[.!?]+', answer) if s.strip()]
        
        if not claims:
            return 1.0
        
        # Combine all context
        context_text = " ".join([doc['content'] for doc in context_docs])
        context_lower = context_text.lower()
        
        supported_claims = 0
        for claim in claims:
            # Simple n-gram overlap check
            claim_words = set(claim.lower().split())
            context_words = set(context_lower.split())
            
            # If >50% of claim words appear in context, consider supported
            if len(claim_words) > 0:
                overlap = len(claim_words.intersection(context_words)) / len(claim_words)
                if overlap > 0.5:
                    supported_claims += 1
        
        faithfulness = supported_claims / len(claims)
        return faithfulness
    
    def faithfulness_llm_judge(self, answer: str, context_docs: List[Dict]) -> Tuple[float, str]:
        """
        Faithfulness using LLM-as-judge (more accurate but requires API call)
        
        Args:
            answer: Generated answer
            context_docs: Retrieved documents
            
        Returns:
            (score, reasoning)
        """
        if not self.llm_client:
            return self.faithfulness_score(answer, context_docs), "Rule-based fallback"
        
        context_text = "\n\n".join([f"Doc {i+1}: {doc['content']}" for i, doc in enumerate(context_docs)])
        
        prompt = f"""Evaluate the FAITHFULNESS of the answer based on the provided context.

Context Documents:
{context_text}

Generated Answer:
{answer}

Rate faithfulness on scale 0-10 where:
- 10: All claims in answer are directly supported by context
- 5: Some claims supported, some unsupported
- 0: Answer contradicts or has no support in context

Respond in format:
Score: [0-10]
Reasoning: [brief explanation]"""
        
        try:
            result = self.llm_client.generate(prompt, context=None)
            response = result['response']
            
            # Extract score
            score_match = re.search(r'Score:\s*(\d+)', response)
            if score_match:
                score = int(score_match.group(1)) / 10.0
                reasoning = response.split('Reasoning:')[-1].strip() if 'Reasoning:' in response else ""
                return score, reasoning
        except Exception as e:
            print(f"LLM judge failed: {e}")
        
        return self.faithfulness_score(answer, context_docs), "LLM judge failed, using fallback"
    
    @staticmethod
    def relevance_score(answer: str, question: str) -> float:
        """
        Relevance: How well answer addresses the question
        Uses lexical overlap and question type matching
        
        Args:
            answer: Generated answer
            question: Original question
            
        Returns:
            Relevance score [0, 1]
        """
        # Extract key terms from question (remove stop words)
        stop_words = {'what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'the', 'a', 'an'}
        question_words = set(question.lower().split()) - stop_words
        answer_words = set(answer.lower().split())
        
        if not question_words:
            return 1.0
        
        # Calculate overlap
        overlap = len(question_words.intersection(answer_words)) / len(question_words)
        
        # Check if answer is too short (likely not addressing question)
        if len(answer.split()) < 5:
            overlap *= 0.5
        
        # Check for question type addressing
        question_lower = question.lower()
        answer_lower = answer.lower()
        
        type_bonus = 0.0
        if question_lower.startswith('how many') or question_lower.startswith('what is the number'):
            # Expect numbers in answer
            if re.search(r'\d+', answer):
                type_bonus = 0.2
        elif question_lower.startswith('yes') or question_lower.startswith('is') or question_lower.startswith('does'):
            # Expect yes/no
            if 'yes' in answer_lower or 'no' in answer_lower:
                type_bonus = 0.2
        
        relevance = min(overlap + type_bonus, 1.0)
        return relevance
    
    def relevance_llm_judge(self, answer: str, question: str) -> Tuple[float, str]:
        """
        Relevance using LLM-as-judge
        
        Args:
            answer: Generated answer
            question: Original question
            
        Returns:
            (score, reasoning)
        """
        if not self.llm_client:
            return self.relevance_score(answer, question), "Rule-based fallback"
        
        prompt = f"""Evaluate how RELEVANT the answer is to the question.

Question: {question}

Answer: {answer}

Rate relevance on scale 0-10 where:
- 10: Answer directly and completely addresses the question
- 5: Answer partially addresses the question
- 0: Answer is off-topic or doesn't address the question

Respond in format:
Score: [0-10]
Reasoning: [brief explanation]"""
        
        try:
            result = self.llm_client.generate(prompt, context=None)
            response = result['response']
            
            score_match = re.search(r'Score:\s*(\d+)', response)
            if score_match:
                score = int(score_match.group(1)) / 10.0
                reasoning = response.split('Reasoning:')[-1].strip() if 'Reasoning:' in response else ""
                return score, reasoning
        except Exception as e:
            print(f"LLM judge failed: {e}")
        
        return self.relevance_score(answer, question), "LLM judge failed, using fallback"


class EndToEndMetrics:
    """End-to-end RAG evaluation metrics"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
    
    def correctness_score(self, answer: str, ground_truth: str) -> float:
        """
        Correctness: How correct the answer is compared to ground truth
        Uses semantic similarity and fact overlap
        
        Args:
            answer: Generated answer
            ground_truth: Reference correct answer
            
        Returns:
            Correctness score [0, 1]
        """
        # Tokenize and normalize
        answer_tokens = set(answer.lower().split())
        truth_tokens = set(ground_truth.lower().split())
        
        if not truth_tokens:
            return 1.0
        
        # F1-style metric
        overlap = answer_tokens.intersection(truth_tokens)
        if not overlap:
            return 0.0
        
        precision = len(overlap) / len(answer_tokens) if answer_tokens else 0
        recall = len(overlap) / len(truth_tokens) if truth_tokens else 0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def correctness_llm_judge(self, answer: str, ground_truth: str, question: str) -> Tuple[float, str]:
        """
        Correctness using LLM-as-judge
        
        Args:
            answer: Generated answer
            ground_truth: Reference answer
            question: Original question
            
        Returns:
            (score, reasoning)
        """
        if not self.llm_client:
            return self.correctness_score(answer, ground_truth), "Rule-based fallback"
        
        prompt = f"""Evaluate the CORRECTNESS of the generated answer against the ground truth.

Question: {question}

Ground Truth Answer: {ground_truth}

Generated Answer: {answer}

Rate correctness on scale 0-10 where:
- 10: Answer is factually correct and complete
- 5: Answer is partially correct
- 0: Answer is incorrect or contradicts ground truth

Respond in format:
Score: [0-10]
Reasoning: [brief explanation]"""
        
        try:
            result = self.llm_client.generate(prompt, context=None)
            response = result['response']
            
            score_match = re.search(r'Score:\s*(\d+)', response)
            if score_match:
                score = int(score_match.group(1)) / 10.0
                reasoning = response.split('Reasoning:')[-1].strip() if 'Reasoning:' in response else ""
                return score, reasoning
        except Exception as e:
            print(f"LLM judge failed: {e}")
        
        return self.correctness_score(answer, ground_truth), "LLM judge failed, using fallback"
    
    @staticmethod
    def hallucination_rate(answer: str, context_docs: List[Dict]) -> Tuple[float, List[str]]:
        """
        Hallucination Rate: Proportion of answer that's not supported by context
        
        Args:
            answer: Generated answer
            context_docs: Retrieved documents
            
        Returns:
            (hallucination_rate, hallucinated_claims)
        """
        # Extract factual claims (sentences with specific facts)
        claims = []
        for sentence in re.split(r'[.!?]+', answer):
            sentence = sentence.strip()
            if sentence and len(sentence.split()) > 3:
                # Consider sentences with numbers, proper nouns, or specific terms as claims
                if re.search(r'\d+|[A-Z][a-z]+', sentence):
                    claims.append(sentence)
        
        if not claims:
            return 0.0, []
        
        # Combine context
        context_text = " ".join([doc['content'] for doc in context_docs]).lower()
        
        hallucinated = []
        for claim in claims:
            # Check if key facts in claim appear in context
            claim_lower = claim.lower()
            
            # Extract key terms (numbers, capitalized words)
            key_terms = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?|\b[A-Z][a-z]+\b', claim)
            
            if key_terms:
                # Check if any key term is in context
                found = any(term.lower() in context_text for term in key_terms)
                if not found:
                    hallucinated.append(claim)
            else:
                # For claims without key terms, use word overlap
                claim_words = set(claim_lower.split())
                context_words = set(context_text.split())
                overlap = len(claim_words.intersection(context_words)) / len(claim_words) if claim_words else 0
                
                if overlap < 0.3:  # Less than 30% overlap = likely hallucinated
                    hallucinated.append(claim)
        
        hallucination_rate = len(hallucinated) / len(claims)
        return hallucination_rate, hallucinated
    
    def hallucination_llm_judge(self, answer: str, context_docs: List[Dict]) -> Tuple[float, str, List[str]]:
        """
        Hallucination detection using LLM-as-judge
        
        Args:
            answer: Generated answer
            context_docs: Retrieved documents
            
        Returns:
            (hallucination_rate, reasoning, hallucinated_claims)
        """
        if not self.llm_client:
            rate, claims = self.hallucination_rate(answer, context_docs)
            return rate, "Rule-based fallback", claims
        
        context_text = "\n\n".join([f"Doc {i+1}: {doc['content']}" for i, doc in enumerate(context_docs)])
        
        prompt = f"""Identify HALLUCINATIONS in the answer - facts not supported by context.

Context Documents:
{context_text}

Generated Answer:
{answer}

Analyze each factual claim in the answer and check if it's supported by the context.

Respond in format:
Hallucination Rate: [0-100]%
Hallucinated Claims: [list any unsupported claims, or "None"]
Reasoning: [brief explanation]"""
        
        try:
            result = self.llm_client.generate(prompt, context=None)
            response = result['response']
            
            # Extract hallucination rate
            rate_match = re.search(r'Hallucination Rate:\s*(\d+)', response)
            hallucination_rate = int(rate_match.group(1)) / 100.0 if rate_match else 0.0
            
            # Extract hallucinated claims
            claims_section = response.split('Hallucinated Claims:')[-1].split('Reasoning:')[0] if 'Hallucinated Claims:' in response else ""
            hallucinated_claims = [c.strip() for c in claims_section.split('\n') if c.strip() and c.strip().lower() != 'none']
            
            reasoning = response.split('Reasoning:')[-1].strip() if 'Reasoning:' in response else ""
            
            return hallucination_rate, reasoning, hallucinated_claims
        except Exception as e:
            print(f"LLM judge failed: {e}")
            rate, claims = self.hallucination_rate(answer, context_docs)
            return rate, "LLM judge failed, using fallback", claims


class RAGEvaluator:
    """Complete RAG evaluation suite"""
    
    def __init__(self, llm_client=None, use_llm_judge: bool = False):
        """
        Initialize RAG evaluator
        
        Args:
            llm_client: LLM client for LLM-as-judge evaluations
            use_llm_judge: Whether to use LLM-based evaluation (more accurate but slower/costly)
        """
        self.retriever_metrics = RetrieverMetrics()
        self.generator_metrics = GeneratorMetrics(llm_client)
        self.e2e_metrics = EndToEndMetrics(llm_client)
        self.use_llm_judge = use_llm_judge and llm_client is not None
    
    def evaluate_retrieval(
        self,
        retrieved_doc_ids: List[str],
        relevant_doc_ids: Set[str],
        relevance_scores: Optional[Dict[str, float]] = None,
        k_values: List[int] = [3, 5, 10]
    ) -> Dict:
        """
        Comprehensive retrieval evaluation
        
        Args:
            retrieved_doc_ids: List of retrieved document IDs (ordered)
            relevant_doc_ids: Set of ground truth relevant doc IDs
            relevance_scores: Optional dict of doc_id -> relevance score for nDCG
            k_values: K values to compute metrics at
            
        Returns:
            Dict of metrics
        """
        results = {}
        
        # Recall@K
        for k in k_values:
            results[f'recall@{k}'] = self.retriever_metrics.recall_at_k(
                retrieved_doc_ids, relevant_doc_ids, k
            )
            results[f'precision@{k}'] = self.retriever_metrics.precision_at_k(
                retrieved_doc_ids, relevant_doc_ids, k
            )
        
        # MRR
        results['mrr'] = self.retriever_metrics.mean_reciprocal_rank(
            retrieved_doc_ids, relevant_doc_ids
        )
        
        # Average Precision
        results['average_precision'] = self.retriever_metrics.average_precision(
            retrieved_doc_ids, relevant_doc_ids
        )
        
        # nDCG (if relevance scores provided)
        if relevance_scores:
            for k in k_values:
                results[f'ndcg@{k}'] = self.retriever_metrics.ndcg_at_k(
                    retrieved_doc_ids, relevance_scores, k
                )
        
        return results
    
    def evaluate_generation(
        self,
        answer: str,
        question: str,
        context_docs: List[Dict]
    ) -> Dict:
        """
        Comprehensive generation evaluation
        
        Args:
            answer: Generated answer
            question: Original question
            context_docs: Retrieved documents used
            
        Returns:
            Dict of metrics
        """
        results = {}
        
        # Faithfulness
        if self.use_llm_judge:
            faith_score, faith_reasoning = self.generator_metrics.faithfulness_llm_judge(
                answer, context_docs
            )
            results['faithfulness'] = faith_score
            results['faithfulness_reasoning'] = faith_reasoning
        else:
            results['faithfulness'] = self.generator_metrics.faithfulness_score(
                answer, context_docs
            )
        
        # Relevance
        if self.use_llm_judge:
            rel_score, rel_reasoning = self.generator_metrics.relevance_llm_judge(
                answer, question
            )
            results['relevance'] = rel_score
            results['relevance_reasoning'] = rel_reasoning
        else:
            results['relevance'] = self.generator_metrics.relevance_score(
                answer, question
            )
        
        return results
    
    def evaluate_end_to_end(
        self,
        answer: str,
        question: str,
        ground_truth: Optional[str],
        context_docs: List[Dict]
    ) -> Dict:
        """
        Complete end-to-end evaluation
        
        Args:
            answer: Generated answer
            question: Original question
            ground_truth: Reference answer (optional)
            context_docs: Retrieved documents
            
        Returns:
            Dict of metrics
        """
        results = {}
        
        # Correctness (if ground truth available)
        if ground_truth:
            if self.use_llm_judge:
                corr_score, corr_reasoning = self.e2e_metrics.correctness_llm_judge(
                    answer, ground_truth, question
                )
                results['correctness'] = corr_score
                results['correctness_reasoning'] = corr_reasoning
            else:
                results['correctness'] = self.e2e_metrics.correctness_score(
                    answer, ground_truth
                )
        
        # Hallucination Rate
        if self.use_llm_judge:
            hall_rate, hall_reasoning, hall_claims = self.e2e_metrics.hallucination_llm_judge(
                answer, context_docs
            )
            results['hallucination_rate'] = hall_rate
            results['hallucination_reasoning'] = hall_reasoning
            results['hallucinated_claims'] = hall_claims
        else:
            hall_rate, hall_claims = self.e2e_metrics.hallucination_rate(
                answer, context_docs
            )
            results['hallucination_rate'] = hall_rate
            results['hallucinated_claims'] = hall_claims
        
        return results
    
    def evaluate_full_pipeline(
        self,
        question: str,
        answer: str,
        retrieved_doc_ids: List[str],
        context_docs: List[Dict],
        relevant_doc_ids: Optional[Set[str]] = None,
        ground_truth: Optional[str] = None,
        relevance_scores: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Evaluate complete RAG pipeline
        
        Returns:
            Comprehensive metrics dict
        """
        results = {
            'question': question,
            'answer': answer,
            'retrieval': {},
            'generation': {},
            'end_to_end': {}
        }
        
        # Retrieval metrics (if ground truth available)
        if relevant_doc_ids:
            results['retrieval'] = self.evaluate_retrieval(
                retrieved_doc_ids,
                relevant_doc_ids,
                relevance_scores
            )
        
        # Generation metrics
        results['generation'] = self.evaluate_generation(
            answer, question, context_docs
        )
        
        # End-to-end metrics
        results['end_to_end'] = self.evaluate_end_to_end(
            answer, question, ground_truth, context_docs
        )
        
        # Overall score (weighted average)
        scores = []
        if 'faithfulness' in results['generation']:
            scores.append(results['generation']['faithfulness'])
        if 'relevance' in results['generation']:
            scores.append(results['generation']['relevance'])
        if 'hallucination_rate' in results['end_to_end']:
            scores.append(1 - results['end_to_end']['hallucination_rate'])
        
        results['overall_score'] = sum(scores) / len(scores) if scores else 0.0
        
        return results