# app.py
import streamlit as st
import re
from pathlib import Path

# Import your existing components
from embeddings.embedder import DocumentEmbedder
from retrieval.retriever import DocumentRetriever
from llm.llm_interface import LLMInterface

# Set page configuration
st.set_page_config(
    page_title="CelcomDigi Knowledge Assistant",
    page_icon="📱",
    layout="centered"
)

# Load RAG components (cached so they don't reload on each interaction)
@st.cache_resource
def load_rag_system():
    try:
        embedder = DocumentEmbedder(persist_directory="chroma_db")
        vectorstore = embedder.load_vector_store()
        retriever = DocumentRetriever(vectorstore)
        llm = LLMInterface(model_name="qwen3:0.6b")
        
        # Use a standard prompt format
        from langchain.prompts import PromptTemplate
        llm.prompt_template = PromptTemplate.from_template(
            """You are an AI assistant for CelcomDigi, a telecommunications company in Malaysia. 
            Answer the following question based ONLY on the provided context.
            Provide a clear, concise answer.
            If you don't know the answer or if it's not in the context, say so.
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:"""
        )
        
        return retriever, llm, True
    except Exception as e:
        st.error(f"Error loading RAG system: {str(e)}")
        return None, None, False

# Function to clean the response of thinking tags and sections
def clean_response(response):
    # Remove <think>...</think> tags
    cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    # Remove **Thinking** or ## Thinking sections
    cleaned = re.sub(r'(?i)(\*\*|\#\#?\s*)Thinking:?.*?(\*\*|\n\n)', '', cleaned, flags=re.DOTALL)
    
    # Remove **Answer** or ## Answer headers (keep the content)
    cleaned = re.sub(r'(?i)(\*\*|\#\#?\s*)Answer:?(\*\*|\s*)', '', cleaned)
    
    # Clean up any extra newlines or spaces
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = cleaned.strip()
    
    return cleaned

# Function to extract thinking from response
def extract_thinking(response):
    # Try to extract <think>...</think> tags
    think_tags = re.search(r'<think>(.*?)</think>', response, flags=re.DOTALL)
    if think_tags:
        return think_tags.group(1).strip()
    
    # Try to extract **Thinking** or ## Thinking sections
    thinking_section = re.search(r'(?i)(\*\*|\#\#?\s*)Thinking:?.*?(\*\*|\n)(.*?)(?=(\*\*|\#\#?\s*)Answer:?|$)', 
                               response, flags=re.DOTALL)
    if thinking_section and thinking_section.group(3):
        return thinking_section.group(3).strip()
    
    # If no explicit thinking section, return empty string
    return ""

# Main app title
st.title("CelcomDigi Knowledge Assistant")

# Load system components
retriever, llm, system_loaded = load_rag_system()

if not system_loaded:
    st.error("Failed to initialize the system. Please check the logs.")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm the CelcomDigi Knowledge Assistant. How can I help you today?"}
    ]

# Display chat history (excluding detailed sections)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Check if this is a structured response
        if isinstance(message["content"], dict) and "answer" in message["content"]:
            # Only show the answer part in the chat history
            st.markdown(message["content"]["answer"])
        else:
            # Regular message
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about CelcomDigi products and services..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Show user message in chat
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching for information..."):
            # 1. FIRST: Retrieve and show sources
            retrieved_docs = retriever.retrieve(prompt, k=3)
            
            st.markdown("### Source Documents")
            sources_content = ""
            if retrieved_docs:
                for i, doc in enumerate(retrieved_docs):
                    with st.container():
                        st.markdown(f"**Source {i+1}**: {doc['metadata'].get('title', 'Unknown')}")
                        st.markdown(f"**Question**: {doc['metadata'].get('question', 'Not available')}")
                        st.markdown(f"**Content**: {doc['content'][:150]}..." if len(doc['content']) > 150 else f"**Content**: {doc['content']}")
                        st.markdown("---")
                        
                        # Save sources for history
                        sources_content += f"Source {i+1}: {doc['metadata'].get('title', 'Unknown')}\n"
                        sources_content += f"Question: {doc['metadata'].get('question', 'Not available')}\n"
                        sources_content += f"Content: {doc['content'][:150]}...\n\n"
            else:
                st.write("No sources found")
                sources_content = "No sources found"
            
            # 2. Generate full response
            full_response = llm.generate_response(prompt, retrieved_docs)
            
            # 3. Extract thinking and clean the answer
            thinking = extract_thinking(full_response)
            clean_answer = clean_response(full_response)
            
            # 4. SECOND: Show thinking if available
            if thinking:
                st.markdown("### Thinking Rationale")
                st.markdown(thinking)
            
            # 5. THIRD: Show the answer
            st.markdown("### Answer")
            st.markdown(clean_answer)
    
    # Add structured response to chat history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": {
            "sources": sources_content,
            "thinking": thinking,
            "answer": clean_answer,
            "full_response": full_response
        }
    })

# Simple footer
st.markdown("---")
st.markdown("© 2025 CelcomDigi RAG Project by BYY")