# document_processing/processor.py
"""
This module handles document processing for the RAG system, including:
- OCR-based text extraction from PDFs
- Text cleaning and normalization
- Document chunking based on question-answer pairs
- Metadata extraction and management
"""

# Standard library imports
import re
import unicodedata
from pathlib import Path

# Third-party imports for OCR and document processing
import pytesseract
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import pypdfium2 as pdfium

# Configure Tesseract OCR path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class DocumentProcessor:
    """
    Main class for processing documents in the RAG system.
    Handles OCR, text extraction, cleaning, and chunking of PDF documents.
    """
    
    def __init__(self, data_dir):
        """Initialize the processor with the data directory path."""
        self.data_dir = data_dir
    
    def load_documents(self, pdf_files):
        """
        Load and process PDF documents using OCR.
        
        Args:
            pdf_files (list): List of PDF filenames to process
            
        Returns:
            list: List of Document objects containing extracted text and metadata
        """
        documents = []
        for pdf_file in pdf_files:
            pdf_path = self.data_dir / pdf_file
            if not pdf_path.exists():
                continue
                
            print(f"Loading {pdf_file} with OCR...")
            
            # Extract text using OCR
            text = self.extract_text_with_ocr(pdf_path)
            
            # Only add documents with non-empty content
            if len(text.strip()) > 0:
                print(f"  Extracted {len(text)} characters with OCR")
                documents.append(Document(
                    page_content=text,
                    metadata={"source": pdf_file}
                ))
        
        return documents
    
    def extract_text_with_ocr(self, pdf_path):
        """
        Extract text from a PDF using OCR technology.
        
        Args:
            pdf_path (Path): Path to the PDF file
            
        Returns:
            str: Extracted text content from all pages
        """
        pdf = pdfium.PdfDocument(pdf_path)
        text_content = ""
        
        for i, page in enumerate(pdf):
            # Convert PDF page to image for OCR processing
            bitmap = page.render(scale=2.0, rotation=0)
            pil_image = bitmap.to_pil()
            
            # Perform OCR on the page image
            page_text = pytesseract.image_to_string(pil_image)
            print(f"    Page {i+1}: OCR extracted {len(page_text)} characters")
            text_content += page_text + "\n\n"
        
        return text_content
    
    def clean_text(self, text):
        """
        Clean and normalize extracted text.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned and normalized text
        """
        # Remove extra whitespace and normalize spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Fix common OCR artifacts (e.g., hyphenated words)
        text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
        
        # Remove page number artifacts
        text = re.sub(r'Page \d+ of \d+', '', text)
        
        # Normalize Unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        return text
    
    def extract_metadata(self, document):
        """
        Extract metadata from document content.
        
        Args:
            document (Document): Document object to process
            
        Returns:
            dict: Extracted metadata including title, date, and description
        """
        # Extract document title using regex patterns
        title_match = re.search(r'(CelcomDigi.*?)(Pass|Offer|Launch|Series)', document.page_content)
        # Examples it would match:
        # - "CelcomDigi Sahur Pass"
        # - "CelcomDigi Raya Offer"
        # - "CelcomDigi Galaxy S25 Series Launch"

        # If first pattern fails, it tries:
        if not title_match:
            title_match = re.search(r'(Port-In\s+Rebate\s+Offer|Samsung\s+Galaxy\s+S\d+\s+Series)', document.page_content)
            # Examples it would match:
            # - "Port-In Rebate Offer"
            # - "Samsung Galaxy S25 Series"
            
        title = title_match.group(0) if title_match else document.metadata.get("source", "Unknown")
        
        # Extract modification date
        date_match = re.search(r'Modified on ([A-Za-z]+,\s*\d+\s*[A-Za-z]+(?:\s*at\s*[\d:]+\s*[AP]M)?)', document.page_content)
        date = date_match.group(1) if date_match else ""
        
        # Extract first question as description
        first_question = ""
        question_match = re.search(r'\d+[\.,]\s+(.*?)[.?]', document.page_content)
        if question_match:
            first_question = question_match.group(1).strip()
        
        return {
            "source": document.metadata.get("source", "Unknown"),
            "title": title,
            "date": date if date else "",
            "description": first_question
        }
    
    def chunk_documents(self, documents):
        """
        Split documents into chunks based on question-answer pairs.
        
        Args:
            documents (list): List of Document objects to chunk
            
        Returns:
            list: List of chunked Document objects
        """
        if not documents:
            return []
        
        print("Chunking documents into question-answer pairs...")
        all_chunks = []
        
        for document in documents:
            text = document.page_content
            metadata = document.metadata.copy()
            
            # Extract and add metadata
            doc_metadata = self.extract_metadata(document)
            metadata.update(doc_metadata)
            
            # Pattern for finding questions in text
            question_pattern = r'(?:\n|\A)\s*(\d+[\.,]?\s*(?:[A-Za-z0-9])[^\n]{2,}?(?:[?:\.]))'
            matches = list(re.finditer(question_pattern, text))
            
            # Fallback patterns if no questions found
            if not matches:
                simple_pattern = r'(?:\n|\A)\s*(\d+[\.,])'
                matches = list(re.finditer(simple_pattern, text))
            
            # Use standard chunking if no question patterns found
            if not matches:
                print(f"  No question-answer pairs found in {metadata['source']}, using standard chunking")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=100,
                    separators=["\n\n", "\n", ". ", " ", ""],
                    length_function=len
                )
                standard_chunks = text_splitter.split_text(text)
                for chunk in standard_chunks:
                    all_chunks.append(Document(
                        page_content=self.clean_text(chunk),
                        metadata=metadata.copy()
                    ))
                continue
            
            # Process question positions and create chunks
            positions = []
            for match in matches:
                q_num_match = re.match(r'\s*(\d+)[\.,]?', match.group(1))
                if q_num_match:
                    q_num = int(q_num_match.group(1))
                    positions.append((match.start(), q_num, match.group(1)))
            
            # Sort by question number
            positions.sort(key=lambda x: x[1])
            
            # Create chunks for each question-answer pair
            for i, (pos, q_num, question) in enumerate(positions):
                end_pos = positions[i+1][0] if i < len(positions) - 1 else len(text)
                chunk_text = text[pos:end_pos].strip()
                chunk_text = self.clean_text(chunk_text)
                
                # Prepare chunk metadata
                chunk_metadata = metadata.copy()
                question_text = re.sub(r'^\s*\d+[\.,]?\s*', '', question).strip()
                chunk_metadata["question"] = question_text
                chunk_metadata["question_number"] = q_num
                
                if chunk_text:
                    all_chunks.append(Document(
                        page_content=chunk_text,
                        metadata=chunk_metadata
                    ))
            
            print(f"  Created {len(positions)} question-answer chunks from {metadata['source']}")
        
        print(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def save_chunks_to_file(self, chunks, filename="document_chunks.txt"):
        """
        Save processed chunks to a text file for inspection.
        
        Args:
            chunks (list): List of Document chunks to save
            filename (str): Output filename
        """
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Total chunks: {len(chunks)}\n\n")
            for i, chunk in enumerate(chunks):
                f.write(f"CHUNK {i+1}\n")
                f.write(f"Source: {chunk.metadata.get('source', 'Unknown')}\n")
                f.write(f"Title: {chunk.metadata.get('title', 'Unknown')}\n")
                if "question" in chunk.metadata:
                    f.write(f"Question: {chunk.metadata.get('question', '')}\n")
                f.write("-" * 80 + "\n")
                f.write(chunk.page_content + "\n")
                f.write("-" * 80 + "\n\n")
        
        print(f"Saved {len(chunks)} chunks to {filename}")
    
    def process_documents(self, pdf_files):
        """
        Main method to process documents through the entire pipeline.
        
        Args:
            pdf_files (list): List of PDF filenames to process
            
        Returns:
            list: Processed and chunked Document objects
        """
        # Load documents with OCR
        documents = self.load_documents(pdf_files)
        print(f"Loaded {len(documents)} documents")
        
        # Split into chunks
        print("Chunking documents...")
        chunks = self.chunk_documents(documents)
        
        # Save chunks for inspection
        self.save_chunks_to_file(chunks)
        
        return chunks