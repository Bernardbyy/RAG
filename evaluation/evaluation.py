# evaluation/evaluation.py
"""
This module provides evaluation tools for the RAG system's retrieval performance.
It implements Recall@k metric and measures retrieval times to assess the system's
effectiveness and efficiency.

Key Features:
- Recall@k evaluation metric
- Retrieval time measurement
- Detailed performance reporting
- Command-line interface for evaluation
"""

# Standard library imports
import os
import sys
import time
from pathlib import Path

# Add the parent directory to the Python path
# This allows importing modules from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required components
from .test_questions import TEST_QUESTIONS  # Predefined test questions
from embeddings.embedder import DocumentEmbedder  # For loading vector store
from retrieval.retriever import DocumentRetriever  # For document retrieval

def evaluate_recall_at_k(k=3):
    """
    Evaluate the RAG system's retrieval performance using Recall@k metric.
    
    This function:
    1. Loads the vector store and retriever
    2. Tests each question in the test set
    3. Measures retrieval accuracy and speed
    4. Calculates and reports overall metrics
    
    Args:
        k (int): Number of documents to retrieve per query
                Defaults to 3 for standard evaluation
    
    Returns:
        dict: Evaluation metrics including:
            - recall_at_k: Proportion of correct retrievals
            - avg_retrieval_time: Average time per retrieval
    
    Note:
        Recall@k measures if the correct document is among the top k results.
        A score of 1.0 means the correct document was always in the top k results.
    """
    # Initialize the evaluation components
    print("Loading vector store...")
    embedder = DocumentEmbedder(persist_directory="chroma_db")
    vectorstore = embedder.load_vector_store()
    retriever = DocumentRetriever(vectorstore)
    
    # Initialize evaluation metrics
    total_questions = len(TEST_QUESTIONS)
    correct_retrievals = 0
    retrieval_times = []
    
    print(f"\nEvaluating Recall@{k} for {total_questions} questions...\n")
    
    # Evaluate each test question
    for i, test_case in enumerate(TEST_QUESTIONS):
        question = test_case["question"]
        source_doc = test_case["source_document"]
        
        print(f"Q{i+1}: {question}")
        
        # Measure retrieval time
        start_retrieval = time.time()
        retrieved_docs = retriever.retrieve(question, k=k)
        retrieval_time = time.time() - start_retrieval
        retrieval_times.append(retrieval_time)
        
        # Check if the correct document was retrieved
        sources = [doc["metadata"].get("source", "Unknown") for doc in retrieved_docs]
        correct_retrieval = source_doc in sources
        
        # Report retrieval results
        if correct_retrieval:
            correct_retrievals += 1
            source_position = sources.index(source_doc) + 1
            print(f"✓ Correct document retrieved (position {source_position})")
        else:
            print(f"✗ Correct document NOT retrieved")
            print(f"  Expected: {source_doc}")
            print(f"  Retrieved: {sources}")
        
        print(f"  Retrieval time: {retrieval_time:.2f}s")
        print("-" * 80)
    
    # Calculate overall performance metrics
    recall_at_k = correct_retrievals / total_questions
    avg_retrieval_time = sum(retrieval_times) / len(retrieval_times)
    
    # Print comprehensive evaluation results
    print("\n====== EVALUATION RESULTS ======")
    print(f"Total questions: {total_questions}")
    print(f"Recall@{k}: {recall_at_k:.2f} ({correct_retrievals}/{total_questions})")
    print(f"Average retrieval time: {avg_retrieval_time:.4f}s")
    
    return {
        "recall_at_k": recall_at_k,
        "avg_retrieval_time": avg_retrieval_time
    }

if __name__ == "__main__":
    """
    Command-line interface for running the evaluation.
    
    Usage:
        python evaluation.py [k]
        
    Args:
        k (optional): Number of documents to retrieve
                     Defaults to 3 if not specified or invalid
    """
    # Set default k value
    k = 3
    
    # Allow command-line override of k
    if len(sys.argv) > 1:
        try:
            k = int(sys.argv[1])
        except ValueError:
            print(f"Invalid k value: {sys.argv[1]}. Using default k=3.")
    
    # Run the evaluation
    evaluate_recall_at_k(k)