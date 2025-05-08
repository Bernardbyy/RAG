# RAG

# Knowledge Assistant Chatbot 

## Overview

CelcomDigi Knowledge Assistant is an AI-powered RAG (Retrieval-Augmented Generation) system designed to provide accurate, context-aware responses to questions about CelcomDigi products and services. This proof-of-concept application demonstrates how AI can efficiently access and leverage information stored within internal documents to deliver intelligent assistance to customer service representatives.

## Key Features

- **Intelligent Document Processing**: Automatically extracts and processes information from PDF documents using OCR technology
- **Question-Answer Based Chunking**: Organizes document content by natural question-answer pairs for optimal retrieval
- **Advanced Semantic Search**: Utilizes specialized question-answering embeddings for precise information retrieval
- **Local LLM Integration**: Runs entirely on-premise using Ollama with Qwen3 models
- **Context-Aware Responses**: Provides responses based on retrieved information with clear source attribution
- **Interactive UI**: Clean, user-friendly interface built with Streamlit
- **Comprehensive Evaluation**: Includes evaluation tools to measure and improve system performance

## System Architecture

## Installation

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com) installed (for local LLM inference)
- Tesseract OCR installed (for scanned document processing)

---

### ⚙️ Setup

1. **Clone this repository:**
   ```yaml
   git clone https://github.com/yourusername/celcomdigi-knowledge-assistant.git
   cd celcomdigi-knowledge-assistant
   ```

2. **Create and activate a virtual environment:**

3. **Install required packages:**
   ```yaml
   pip install -r requirements.txt
   ```

4. **Install Ollama:**
   Follow instructions at [https://ollama.com](https://ollama.com)

   Then pull the required model:
   ```yaml
   ollama pull qwen3:0.6b
   ```

5. **Set up Tesseract OCR:**
   - **Windows**: [Download from UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)

6. **Create a `data/` directory and add your PDFs:**
   ```yaml
   mkdir -p data
   # Copy your PDF files into the data directory
   ```

---

## 🧠 Usage

### ▶️ Running the Application

Start the Streamlit interface:
```yaml
streamlit run app.py
```

> The web interface will open in your browser (usually at [http://localhost:8501](http://localhost:8501)).

Ask questions about CelcomDigi products and services in natural language.

---

### 🔄 Rebuilding the Vector Store

If you add new documents or want to refresh embeddings:
```yaml
python main.py
```

---

### 📈 Running Evaluation

Evaluate the system's retrieval performance:
```yaml
python -m evaluation.evaluation
```

To test different `k` values for top-k retrieval:
```yaml
python -m evaluation.evaluation 5
```
