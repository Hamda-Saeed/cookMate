"""
Baseline Model: LLM without RAG
This model generates answers without access to the recipe knowledge base
"""

from cookmate_rag import GroqLLM
from typing import Dict, Optional


class BaselineModel:
    """
    Baseline model that uses only the LLM without RAG retrieval
    This simulates a model without access to the recipe knowledge base
    """
    
    def __init__(self, model: str = "llama-3.1-8b-instant"):
        """Initialize baseline model with LLM only"""
        print("Initializing Baseline Model (No RAG)...")
        self.llm = GroqLLM(model=model)
        print("Baseline Model ready!\n")
    
    def generate_answer(self, question: str) -> str:
        """
        Generate answer using LLM only (no RAG context)
        
        Args:
            question: User question
            
        Returns:
            Generated answer without recipe context
        """
        try:
            # Use LLM without any retrieved context
            # The LLM's generate method takes (prompt, context)
            response = self.llm.generate(question, context="")
            
            if not response or len(response) < 10:
                return "I don't have specific information about that recipe in my knowledge base."
            
            return response
        except Exception as e:
            # Fallback response
            return f"I don't have access to specific recipe information. Please refer to the recipe documentation for accurate details."
    
    def batch_generate(self, questions: list) -> list:
        """
        Generate answers for multiple questions
        
        Args:
            questions: List of question strings
            
        Returns:
            List of generated answers
        """
        answers = []
        for question in questions:
            answer = self.generate_answer(question)
            answers.append(answer)
        return answers

