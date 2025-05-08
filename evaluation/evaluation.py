# evaluation/evaluation.py
import os
import sys
import time
from pathlib import Path

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .test_questions import TEST_QUESTIONS
from embeddings.embedder import DocumentEmbedder
from retrieval.retriever import DocumentRetriever
from llm.llm_interface import LLMInterface

def evaluate_recall_at_k(k=3):
    """Evaluate retrieval performance using Recall@k."""
    # Initialize components
    print("Loading vector store...")
    embedder = DocumentEmbedder(persist_directory="chroma_db")
    vectorstore = embedder.load_vector_store()
    retriever = DocumentRetriever(vectorstore)
    llm = LLMInterface(model_name="qwen3:0.6b")
    
    total_questions = len(TEST_QUESTIONS)
    correct_retrievals = 0
    answer_accuracy = 0
    retrieval_times = []
    
    print(f"\nEvaluating Recall@{k} for {total_questions} questions...\n")
    
    results = []
    
    # Evaluate each question
    for i, test_case in enumerate(TEST_QUESTIONS):
        question = test_case["question"]
        expected_answer = test_case["expected_answer"].lower()
        source_doc = test_case["source_document"]
        
        print(f"Q{i+1}: {question}")
        
        # Time the retrieval process
        start_retrieval = time.time()
        retrieved_docs = retriever.retrieve(question, k=k)
        retrieval_time = time.time() - start_retrieval
        retrieval_times.append(retrieval_time)
        
        # Check if source document was retrieved
        sources = [doc["metadata"].get("source", "Unknown") for doc in retrieved_docs]
        correct_retrieval = source_doc in sources
        
        if correct_retrieval:
            correct_retrievals += 1
            source_position = sources.index(source_doc) + 1
            print(f"✓ Correct document retrieved (position {source_position})")
        else:
            print(f"✗ Correct document NOT retrieved")
            print(f"  Expected: {source_doc}")
            print(f"  Retrieved: {sources}")
        
        # Check if expected answer found in retrieved content
        answer_found = False
        for doc in retrieved_docs:
            if expected_answer.lower() in doc["content"].lower():
                answer_found = True
                break
        
        if answer_found:
            print(f"✓ Expected answer found in retrieved content")
            answer_accuracy += 1
        else:
            print(f"✗ Expected answer NOT found in retrieved content")
            
        print(f"  Retrieval time: {retrieval_time:.2f}s")
        print("-" * 80)
        
        # Save result
        results.append({
            "question": question,
            "expected_answer": expected_answer,
            "source_document": source_doc,
            "retrieval_correct": correct_retrieval,
            "answer_found_in_retrieval": answer_found,
            "retrieval_time": retrieval_time,
            "retrieved_sources": sources
        })
    
    # Calculate overall metrics
    retrieval_accuracy = correct_retrievals / total_questions
    answer_in_context_rate = sum(r["answer_found_in_retrieval"] for r in results) / total_questions
    avg_retrieval_time = sum(retrieval_times) / len(retrieval_times)
    
    # Print overall results
    print("\n====== EVALUATION RESULTS ======")
    print(f"Total questions: {total_questions}")
    print(f"Recall@{k}: {retrieval_accuracy:.2f} ({correct_retrievals}/{total_questions})")
    print(f"Answer in context rate: {answer_in_context_rate:.2f}")
    print(f"Average retrieval time: {avg_retrieval_time:.4f}s")
    
    return {
        "recall_at_k": retrieval_accuracy,
        "answer_in_context_rate": answer_in_context_rate,
        "avg_retrieval_time": avg_retrieval_time,
        "detailed_results": results
    }

if __name__ == "__main__":
    # Default to k=3, but allow command line override
    k = 3
    if len(sys.argv) > 1:
        try:
            k = int(sys.argv[1])
        except ValueError:
            print(f"Invalid k value: {sys.argv[1]}. Using default k=3.")
    
    evaluate_recall_at_k(k)