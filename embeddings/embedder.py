# embeddings/embedder.py
'''
handles the creation and management of vector embeddings for document chunks, 
using Hugging Face's sentence transformers and ChromaDB as the vector database.
'''
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

class DocumentEmbedder:
    def __init__(self, persist_directory="chroma_db"):
        """Set up the document embedder."""
        self.persist_directory = persist_directory
        
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Make sure storage directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
    
    def create_vector_store(self, documents):
        """Create and save a vector database from documents."""
        print(f"Creating vector store with {len(documents)} documents...")
        
        # Clean metadata to remove None values
        for doc in documents:
            # Replace None values with empty strings
            for key in list(doc.metadata.keys()):
                if doc.metadata[key] is None:
                    doc.metadata[key] = ""
                # Convert lists to strings if present
                elif isinstance(doc.metadata[key], list):
                    doc.metadata[key] = str(doc.metadata[key])
        
        # Create the database with persist_directory directly
        # This will automatically persist it
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory
        )
        
        print(f"Vector store created in {self.persist_directory}")
        
        return vectorstore
    
    def load_vector_store(self):
        """Load an existing vector database."""
        if not os.path.exists(self.persist_directory):
            raise ValueError(f"Vector store not found at {self.persist_directory}")
        
        print(f"Loading vector store from {self.persist_directory}...")
        
        # Load from disk
        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model
        )
        
        return vectorstore