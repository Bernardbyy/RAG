# retrieval/retriever.py
'''
This class is responsible for finding relevant documents based on user queries
It performs similarity searches to find the most relevant document chunks for a given query.
'''
from langchain_chroma import Chroma

class DocumentRetriever:
    def __init__(self, vectorstore):
        """Set up the retriever with a vector database."""
        self.vectorstore = vectorstore
    
    def retrieve(self, query, k=3):
        """Find relevant documents for a given query."""
        print(f"Retrieving documents for query: {query}")
        
        # Suppress warnings about relevance scores
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Search for similar documents in the vector store
            results = self.vectorstore.similarity_search_with_relevance_scores(
                query=query,
                k=k
            )
        
        # Format the results into a standard structure
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": score
            })
        
        print(f"Retrieved {len(formatted_results)} documents")
        return formatted_results