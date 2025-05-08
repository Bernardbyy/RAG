# main.py
'''
The main.py file orchestrates your entire RAG system. It initializes all components, processes documents, 
builds the vector store, and provides an interactive interface for users to ask questions about the 
CelcomDigi documents.
'''
import os
from pathlib import Path
import shutil

from document_processing.processor import DocumentProcessor
from embeddings.embedder import DocumentEmbedder
from retrieval.retriever import DocumentRetriever
from llm.llm_interface import LLMInterface

def main():
    print("CelcomDigi Knowledge Assistant")
    print("------------------------------")
    
    # Set up paths and files
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    pdf_files = [
        "celcomdigi-eratkanikatan-sahur-moreh-pass.pdf",
        "celcomdigi_raya_video_internet_pass.pdf",
        "celcomdigi_samsung_galaxy_s25_series_launch.pdf",
        "celcomdigi_port-in-rebate-offer.pdf"
    ]

    # Verify PDF files exist
    for pdf_file in pdf_files:
        if not (data_dir / pdf_file).exists():
            print(f"Warning: {pdf_file} not found")
    
    # Initialize components
    processor = DocumentProcessor(data_dir)
    embedder = DocumentEmbedder(persist_directory="chroma_db")
    
    # Rebuild vector store
    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")

    # Process documents
    print("Processing documents...")
    chunks = processor.process_documents(pdf_files)
    
    # Create vector store
    print(f"Creating vector store with {len(chunks)} chunks...")
    vectorstore = embedder.create_vector_store(chunks)
    
    # Initialize retriever and LLM
    retriever = DocumentRetriever(vectorstore)
    llm = LLMInterface(model_name="mistral:7b")
    
    print("\nKnowledge Assistant is ready. Type 'exit' to quit.")
    
    # Interactive question-answering loop
    while True:
        query = input("\nEnter your question: ")
        if query.lower() == 'exit':
            break
        
        # Retrieve and respond
        retrieved_docs = retriever.retrieve(query, k=4)
        response = llm.generate_response(query, retrieved_docs)
        
        print("\nAnswer:")
        print(response)

if __name__ == "__main__":
    main()