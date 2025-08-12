# llm/llm_interface.py
"""
This module provides an interface for interacting with language models in the RAG system.
It handles the formatting of prompts and generation of responses using Ollama-hosted models.

Key Features:
- Integration with Ollama for local LLM inference
- Custom prompt templating for RAG applications
- Context-aware response generation
- Support for both direct and reasoning-based responses
"""

# Third-party imports for LLM interaction
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

class LLMInterface:
    """
    Interface for interacting with language models in the RAG system.
    
    This class is responsible for:
    1. Managing connections to Ollama-hosted language models
    2. Formatting prompts with context and questions
    3. Generating responses based on retrieved documents
    4. Handling different types of responses (direct vs. reasoning)
    """
    
    def __init__(self, model_name="qwen3:0.6b"):
        """
        Initialize the LLM interface with a specific model.
        
        Args:
            model_name (str): Name of the Ollama model to use
                            Defaults to "qwen3:0.6b" for a good balance of performance and resource usage
        """
        # Initialize connection to Ollama with specified model
        # Ollama must be running locally with the model pulled
        self.llm = OllamaLLM(model=model_name)
        
        # Define the prompt template for RAG responses
        # This template:
        # 1. Sets the AI's role and context
        # 2. Provides instructions for different question types
        # 3. Structures the context and question format
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
        """
        Generate a response based on the query and retrieved documents.
        
        This method:
        1. Formats the retrieved documents into a context string
        2. Creates a prompt with the context and query
        3. Sends the prompt to the LLM
        4. Returns the generated response
        
        Args:
            query (str): The user's question
            retrieved_documents (list): List of dictionaries containing:
                - content: The document text
                - metadata: Document metadata including title
        
        Returns:
            str: The LLM's generated response
        
        Note:
            The response may include:
            - Direct answers for simple questions
            - Step-by-step reasoning for complex questions
            - Source attribution from the context
        """
        # Combine retrieved documents into a structured context
        # Each document is numbered and includes its source
        context_text = ""
        for i, doc in enumerate(retrieved_documents):
            context_text += f"Document {i+1}:\n{doc['content']}\nSource: {doc['metadata'].get('title', 'Unknown')}\n\n"
        
        # Format the complete prompt using the template
        # This combines the context and question in the predefined structure
        prompt = self.prompt_template.format(
            context=context_text,
            question=query
        )
        
        # Send the prompt to the LLM and get the response
        # The LLM will use the context to generate an appropriate answer
        response = self.llm.invoke(prompt)
        
        return response