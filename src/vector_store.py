import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import faiss

class VectorStore:
    """Hybrid vector store using FAISS for dense retrieval and BM25 for sparse"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
        self.index = None
        
    def load_documents(self, filepath: str):
        """Load documents from JSON file"""
        with open(filepath, 'r') as f:
            self.documents = json.load(f)
        
        # Create embeddings
        texts = [doc['content'] for doc in self.documents]
        self.embeddings = self.encoder.encode(texts, show_progress_bar=True)
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"Loaded {len(self.documents)} documents into vector store")
    
    def semantic_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Dense retrieval using embeddings"""
        query_embedding = self.encoder.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            doc = self.documents[idx].copy()
            doc['score'] = float(1 / (1 + distance))  # Convert distance to similarity
            results.append(doc)
        
        return results
    
    def keyword_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Sparse retrieval using simple keyword matching (BM25-like)"""
        query_terms = set(query.lower().split())
        
        scores = []
        for doc in self.documents:
            doc_terms = set(doc['content'].lower().split())
            # Simple overlap score
            overlap = len(query_terms.intersection(doc_terms))
            scores.append(overlap / len(query_terms) if query_terms else 0)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self.documents[idx].copy()
                doc['score'] = float(scores[idx])
                results.append(doc)
        
        return results
    
    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.7) -> List[Dict]:
        """
        Hybrid search combining semantic and keyword search
        alpha: weight for semantic search (1-alpha for keyword)
        """
        semantic_results = self.semantic_search(query, top_k=top_k)
        keyword_results = self.keyword_search(query, top_k=top_k)
        
        # Combine results with weighted scores
        combined = {}
        
        for doc in semantic_results:
            doc_id = doc['id']
            combined[doc_id] = {
                'doc': doc,
                'score': alpha * doc['score']
            }
        
        for doc in keyword_results:
            doc_id = doc['id']
            if doc_id in combined:
                combined[doc_id]['score'] += (1 - alpha) * doc['score']
            else:
                combined[doc_id] = {
                    'doc': doc,
                    'score': (1 - alpha) * doc['score']
                }
        
        # Sort by combined score
        sorted_results = sorted(
            combined.values(),
            key=lambda x: x['score'],
            reverse=True
        )[:top_k]
        
        # Return documents with updated scores
        return [
            {**item['doc'], 'score': item['score']} 
            for item in sorted_results
        ]