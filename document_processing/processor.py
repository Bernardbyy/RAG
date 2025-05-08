# document_processing/processor.py
import re
import unicodedata
from pathlib import Path
import pytesseract
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import pypdfium2 as pdfium

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class DocumentProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
    
    def load_documents(self, pdf_files):
        """Load documents using OCR for image-based PDFs."""
        documents = []
        for pdf_file in pdf_files:
            pdf_path = self.data_dir / pdf_file
            if not pdf_path.exists():
                continue
                
            print(f"Loading {pdf_file} with OCR...")
            
            # Extract text with OCR
            text = self.extract_text_with_ocr(pdf_path)
            
            if len(text.strip()) > 0:
                print(f"  Extracted {len(text)} characters with OCR")
                documents.append(Document(
                    page_content=text,
                    metadata={"source": pdf_file}
                ))
        
        return documents
    
    def extract_text_with_ocr(self, pdf_path):
        """Extract text from a PDF using OCR."""
        pdf = pdfium.PdfDocument(pdf_path)
        text_content = ""
        
        for i, page in enumerate(pdf):
            # Convert PDF page to image
            bitmap = page.render(scale=2.0, rotation=0)
            pil_image = bitmap.to_pil()
            
            # Extract text using OCR
            page_text = pytesseract.image_to_string(pil_image)
            print(f"    Page {i+1}: OCR extracted {len(page_text)} characters")
            text_content += page_text + "\n\n"
        
        return text_content
    
    def clean_text(self, text):
        """Clean extracted text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Fix common OCR artifacts
        text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
        
        # Remove header/footer artifacts
        text = re.sub(r'Page \d+ of \d+', '', text)
        
        # Normalize Unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        return text
    
    def extract_metadata(self, document):
        """Extract metadata from document."""
        # Extract document title
        title_match = re.search(r'(CelcomDigi.*?)(Pass|Offer|Launch|Series)', document.page_content)
        if not title_match:
            title_match = re.search(r'(Port-In\s+Rebate\s+Offer|Samsung\s+Galaxy\s+S\d+\s+Series)', document.page_content)
        
        title = title_match.group(0) if title_match else document.metadata.get("source", "Unknown")
        
        # Extract date
        date_match = re.search(r'Modified on ([A-Za-z]+,\s*\d+\s*[A-Za-z]+(?:\s*at\s*[\d:]+\s*[AP]M)?)', document.page_content)
        date = date_match.group(1) if date_match else ""
        
        # Extract first header as description
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
        """Split documents into chunks based on FAQ question-answer pairs with improved pattern matching."""
        if not documents:
            return []
        
        print("Chunking documents into question-answer pairs...")
        
        all_chunks = []
        
        for document in documents:
            text = document.page_content
            
            # Initialize metadata with document metadata
            metadata = document.metadata.copy()
            
            # Extract and add additional metadata
            doc_metadata = self.extract_metadata(document)
            metadata.update(doc_metadata)
            
            # Find all questions using a more robust regex pattern
            # This pattern accounts for OCR artifacts and various question formats
            question_pattern = r'(?:\n|\A)\s*(\d+[\.,]?\s*(?:[A-Za-z0-9])[^\n]{2,}?(?:[?:\.]))'
            
            # Get all matches
            matches = list(re.finditer(question_pattern, text))
            
            # If no questions found, fall back to looking for numbered items
            if not matches:
                simple_pattern = r'(?:\n|\A)\s*(\d+[\.,])'
                matches = list(re.finditer(simple_pattern, text))
            
            # If still no matches, use standard chunking
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
            
            # Get the starting positions and question numbers
            positions = []
            for match in matches:
                # Extract the question number
                q_num_match = re.match(r'\s*(\d+)[\.,]?', match.group(1))
                if q_num_match:
                    q_num = int(q_num_match.group(1))
                    positions.append((match.start(), q_num, match.group(1)))
            
            # Sort positions by question number to handle out-of-order matches
            positions.sort(key=lambda x: x[1])
            
            # Extract chunks based on question positions
            for i, (pos, q_num, question) in enumerate(positions):
                # Determine the end position (next question or end of text)
                if i < len(positions) - 1:
                    end_pos = positions[i+1][0]
                else:
                    end_pos = len(text)
                
                # Extract the chunk text (question and its answer)
                chunk_text = text[pos:end_pos].strip()
                
                # Clean the chunk text
                chunk_text = self.clean_text(chunk_text)
                
                # Add the question to chunk metadata
                chunk_metadata = metadata.copy()
                
                # Clean up the question text
                question_text = question
                # Remove the question number
                question_text = re.sub(r'^\s*\d+[\.,]?\s*', '', question_text).strip()
                chunk_metadata["question"] = question_text
                chunk_metadata["question_number"] = q_num
                
                # Create and add the chunk
                if chunk_text:
                    all_chunks.append(Document(
                        page_content=chunk_text,
                        metadata=chunk_metadata
                    ))
            
            print(f"  Created {len(positions)} question-answer chunks from {metadata['source']}")
        
        print(f"Total chunks created: {len(all_chunks)}")
        
        return all_chunks
    
    def save_chunks_to_file(self, chunks, filename="document_chunks.txt"):
        """Save document chunks to a text file for inspection."""
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
        """Process documents: load, add metadata, clean, and chunk."""
        # Load documents with OCR
        documents = self.load_documents(pdf_files)
        print(f"Loaded {len(documents)} documents")
        
        # Split into chunks based on question-answer pairs
        print("Chunking documents...")
        chunks = self.chunk_documents(documents)
        
        # Save chunks to file for inspection
        self.save_chunks_to_file(chunks)
        
        return chunks