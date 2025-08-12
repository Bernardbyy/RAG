# embeddings/embedder.py
"""
This module handles the creation and management of vector embeddings for document chunks.
It uses Hugging Face's sentence transformers for generating embeddings and ChromaDB as the vector database.

Key Features:
- Uses multi-qa-MiniLM-L6-cos-v1 model optimized for question-answering
- Supports persistent storage of embeddings
- Handles metadata cleaning and normalization
- Provides both creation and loading of vector stores
"""

# Standard library imports
import os

# Third-party imports for embeddings and vector storage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

class DocumentEmbedder:
    """
    A class to handle document embedding and vector store management.
    
    This class is responsible for:
    1. Converting text documents into vector embeddings
    2. Creating and managing a vector database
    3. Handling persistence of embeddings
    4. Loading existing vector stores
    """
    
    def __init__(self, persist_directory="chroma_db"):
        """
        Initialize the document embedder.
        
        Args:
            persist_directory (str): Directory where vector store will be saved/loaded
                                    Defaults to "chroma_db"
        """
        self.persist_directory = persist_directory
        
        # Initialize the embedding model
        # Using multi-qa-MiniLM-L6-cos-v1 which is optimized for question-answering
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
            model_kwargs={'device': 'cpu'},  # Force CPU usage
            encode_kwargs={'normalize_embeddings': True}  # Normalize embeddings for better similarity search
        )
        
        # Ensure the storage directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
    
    def create_vector_store(self, documents):
        """
        Create and save a vector database from documents.
        
        This method:
        1. Cleans document metadata
        2. Creates embeddings for each document
        3. Stores them in a ChromaDB vector store
        4. Persists the store to disk
        
        Args:
            documents (list): List of Document objects to embed and store
            
        Returns:
            Chroma: The created vector store instance
        """
        print(f"Creating vector store with {len(documents)} documents...")
        
        # Clean and normalize metadata
        for doc in documents:
            # Replace None values with empty strings to avoid serialization issues
            for key in list(doc.metadata.keys()):
                if doc.metadata[key] is None:
                    doc.metadata[key] = ""
                # Convert lists to strings for compatibility
                elif isinstance(doc.metadata[key], list):
                    doc.metadata[key] = str(doc.metadata[key])
        
        # Create and persist the vector store
        # ChromaDB will automatically handle the persistence
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory
        )
        
        print(f"Vector store created in {self.persist_directory}")
        
        return vectorstore
    
    def load_vector_store(self):
        """
        Load an existing vector database from disk.
        
        This method:
        1. Verifies the vector store exists
        2. Loads the store with the same embedding model
        3. Returns the loaded store for querying
        
        Returns:
            Chroma: The loaded vector store instance
            
        Raises:
            ValueError: If the vector store directory doesn't exist
        """
        # Check if vector store exists
        if not os.path.exists(self.persist_directory):
            raise ValueError(f"Vector store not found at {self.persist_directory}")
        
        print(f"Loading vector store from {self.persist_directory}...")
        
        # Load the existing vector store
        # Uses the same embedding model for consistency
        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model
        )
        
        return vectorstore