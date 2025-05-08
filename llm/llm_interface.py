# llm/llm_interface.py
'''
This class manages the interaction with the language model, 
which generates responses based on user queries and retrieved documents in your RAG application.
It formats the context from retrieved documents and user queries into prompts, 
then gets responses from a locally hosted language model using Ollama.
'''

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

class LLMInterface:
    def __init__(self, model_name="qwen3:0.6b"):
        """Set up the LLM interface with Ollama."""
        # Connect to Ollama with the specified model
        self.llm = OllamaLLM(model=model_name)
        
        # Create the RAG prompt template
        self.prompt_template = PromptTemplate.from_template(
            """You are an AI assistant for CelcomDigi, a telecommunications company in Malaysia. 
            Answer the following question based ONLY on the provided context.
            
            For complex questions that require reasoning, use your thinking mode to work through the problem step by step.
            For straightforward questions, provide direct answers.
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:"""
        )
    
    def generate_response(self, query, retrieved_documents):
        """Create a response using the query and retrieved documents."""
        # Combine all retrieved documents into context text
        context_text = ""
        for i, doc in enumerate(retrieved_documents):
            context_text += f"Document {i+1}:\n{doc['content']}\nSource: {doc['metadata'].get('title', 'Unknown')}\n\n"
        
        # Create the full prompt with context and question
        prompt = self.prompt_template.format(
            context=context_text,
            question=query
        )
        
        # Get response from the LLM
        response = self.llm.invoke(prompt)
        
        return response