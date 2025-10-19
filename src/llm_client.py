import os
from typing import List, Dict, Optional
import time
from openai import OpenAI
import re

class LLMClient:
    """
    OpenAI GPT-4 client with calculation and multi-context support
    """
    
    def __init__(
        self, 
        api_key: str,
        model: str = "gpt-4",
        temperature: float = 0.5,
        max_tokens: int = 1000
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key)
        self.enable_calculations = True
        self.multi_context = True
        
    def generate(self, prompt: str, context: List[Dict] = None) -> Dict:
        """
        Generate response using GPT-4
        
        Args:
            prompt: User query
            context: Retrieved documents
            
        Returns:
            Dict with response, tokens, latency, etc.
        """
        start_time = time.time()
        
        # Build full prompt with context
        full_prompt = self._build_prompt(prompt, context)
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            answer = response.choices[0].message.content
            
            # Extract calculations if present
            calculations = self._extract_calculations(answer) if self.enable_calculations else None
            
            latency = time.time() - start_time
            
            return {
                "response": answer,
                "model": self.model,
                "tokens": response.usage.total_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "latency": latency,
                "context_docs": len(context) if context else 0,
                "calculations": calculations,
                "finish_reason": response.choices[0].finish_reason
            }
            
        except Exception as e:
            return {
                "response": f"Error generating response: {str(e)}",
                "model": self.model,
                "tokens": 0,
                "latency": time.time() - start_time,
                "context_docs": 0,
                "error": str(e)
            }
    
    def _get_system_prompt(self) -> str:
        """Get system prompt based on settings"""
        base_prompt = """You are an advanced AI assistant with expertise in analyzing documents and answering questions accurately."""
        
        if self.enable_calculations:
            base_prompt += """

CALCULATION ABILITIES:
- You can perform mathematical calculations and data analysis
- When asked to calculate, show your work step-by-step
- Format calculations clearly using markdown
- Verify results and show units where applicable
- Extract numbers from documents and compute as needed"""
        
        if self.multi_context:
            base_prompt += """

MULTI-CONTEXT ANALYSIS:
- Synthesize information from multiple documents
- Cross-reference facts across sources
- Identify patterns and relationships between documents
- Provide comprehensive answers that integrate multiple perspectives
- Cite which document each piece of information comes from"""
        
        base_prompt += """

RESPONSE GUIDELINES:
- Base answers on the provided context documents
- If information isn't in the context, explicitly state that
- Be precise and factual
- Structure responses clearly with headers if needed
- Use markdown formatting for readability"""
        
        return base_prompt
    
    def _build_prompt(self, query: str, context: List[Dict] = None) -> str:
        """Build comprehensive prompt with multi-document context"""
        if not context:
            return query
        
        # Group documents by source if multi-context enabled
        if self.multi_context and len(context) > 1:
            context_sections = []
            for i, doc in enumerate(context, 1):
                source = doc.get('metadata', {}).get('source', 'Unknown')
                score = doc.get('score', 0)
                
                section = f"""
--- Document {i} ---
**Title:** {doc['title']}
**Source:** {source}
**Relevance Score:** {score:.3f}
**Category:** {doc.get('category', 'N/A')}

**Content:**
{doc['content']}
---
"""
                context_sections.append(section)
            
            context_str = "\n".join(context_sections)
            
            prompt = f"""You have access to {len(context)} documents to answer the user's question. Analyze all documents and provide a comprehensive answer.

{context_str}

**User Question:** {query}

**Instructions:**
- Synthesize information from all relevant documents
- When referencing information, cite the document number (e.g., "According to Document 2...")
- If performing calculations, extract numbers from documents and show your work
- Provide a well-structured, complete answer"""
            
        else:
            # Single context or simple mode
            context_str = "\n\n".join([
                f"**{doc['title']}** (score: {doc.get('score', 0):.3f})\n{doc['content']}"
                for doc in context
            ])
            
            prompt = f"""Context information:
{context_str}

Question: {query}

Answer based on the context provided. If calculations are needed, show your work clearly."""
        
        return prompt
    
    def _extract_calculations(self, text: str) -> Optional[Dict]:
        """Extract calculation results from response"""
        calculations = {}
        
        # Look for calculation patterns
        calc_patterns = [
            r'(?:sum|total|result).*?[:=]\s*([0-9,]+\.?[0-9]*)',
            r'([0-9,]+\.?[0-9]*)\s*(?:total|sum)',
            r'calculated?\s*(?:as|to be)?\s*([0-9,]+\.?[0-9]*)',
        ]
        
        for pattern in calc_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                calculations['extracted_values'] = [m.replace(',', '') for m in matches]
                break
        
        return calculations if calculations else None
    
    def stream_generate(self, prompt: str, context: List[Dict] = None):
        """Stream generation for real-time responses"""
        full_prompt = self._build_prompt(prompt, context)
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def validate_api_key(self) -> bool:
        """Test if API key is valid"""
        try:
            self.client.models.list()
            return True
        except Exception:
            return False