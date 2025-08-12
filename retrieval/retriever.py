# retrieval/retriever.py
"""
This module handles document retrieval in the RAG system.
It performs semantic similarity searches to find the most relevant document chunks
for a given user query using ChromaDB's vector search capabilities.

Key Features:
- Semantic similarity search using vector embeddings
- Relevance scoring for retrieved documents
- Standardized result formatting
- Configurable number of results (k)
"""

# Third-party imports for vector search
from langchain_chroma import Chroma

class DocumentRetriever:
    """
    A class to handle document retrieval using vector similarity search.
    
    This class is responsible for:
    1. Performing semantic similarity searches
    2. Scoring and ranking retrieved documents
    3. Formatting results in a consistent structure
    4. Managing the vector store connection
    """
    
    def __init__(self, vectorstore):
        """
        Initialize the document retriever with a vector store.
        
        Args:
            vectorstore (Chroma): An initialized ChromaDB vector store instance
                                 containing document embeddings
        """
        self.vectorstore = vectorstore
    
    def retrieve(self, query, k=3):
        """
        Find the most relevant documents for a given query.
        
        This method:
        1. Performs a similarity search in the vector store
        2. Scores each result based on relevance
        3. Formats the results into a standard structure
        4. Returns the top k most relevant documents
        
        Args:
            query (str): The user's question or search query
            k (int): Number of documents to retrieve
                    Defaults to 3 for a good balance of context and relevance
        
        Returns:
            list: List of dictionaries containing:
                - content: The document text
                - metadata: Document metadata (source, title, etc.)
                - relevance_score: Similarity score between query and document
                
        Note:
            The relevance score is a float between 0 and 1, where:
            - 1.0 indicates perfect similarity
            - 0.0 indicates no similarity
            Higher scores mean more relevant documents
        """
        print(f"Retrieving documents for query: {query}")
        
        # Suppress warnings about relevance scores
        # These warnings are common with ChromaDB and don't affect functionality
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Perform similarity search with relevance scoring
            # This uses the vector embeddings to find semantically similar documents
            results = self.vectorstore.similarity_search_with_relevance_scores(
                query=query,
                k=k  # Number of results to return
            )
        
        # Format the results into a standardized structure
        # This makes it easier to process the results in other parts of the system
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,      # The actual document text
                "metadata": doc.metadata,         # Document metadata (source, title, etc.)
                "relevance_score": score          # Similarity score
            })
        
        print(f"Retrieved {len(formatted_results)} documents")
        return formatted_results