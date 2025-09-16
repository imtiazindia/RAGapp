# document_processor.py
import os
import tempfile
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import SUPPORTED_EXTENSIONS, TEXT_SPLITTER_CONFIG

class DocumentProcessor:
    """Handle document loading and processing"""
    
    def __init__(self):
        self.loader_map = {
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.pptx': UnstructuredPowerPointLoader,
            '.xlsx': UnstructuredExcelLoader
        }
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=TEXT_SPLITTER_CONFIG['chunk_size'],
            chunk_overlap=TEXT_SPLITTER_CONFIG['chunk_overlap']
        )
    
    def load_documents(self, files) -> List[Document]:
        """Load documents from uploaded files"""
        documents = []
        
        for file in files:
            # Get file extension
            ext = os.path.splitext(file.name)[-1].lower()
            
            if ext not in self.loader_map:
                raise ValueError(f"Unsupported file type: {ext}")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Load document based on file type
                loader = self.loader_map[ext](tmp_file_path)
                loaded_docs = loader.load()
                documents.extend(loaded_docs)
            except Exception as e:
                raise Exception(f"Error processing {file.name}: {str(e)}")
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        return self.text_splitter.split_documents(documents)
