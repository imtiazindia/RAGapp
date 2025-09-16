# embedding_manager.py
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import os
import tempfile
import shutil

from config import OPENAI_CONFIG

class EmbeddingManager:
    """Manage embeddings and vector store using ChromaDB"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=OPENAI_CONFIG['embedding_model'],
            openai_api_key=OPENAI_CONFIG['api_key'] or os.environ.get("OPENAI_API_KEY")
        )
        self.vector_store = None
        self.persist_directory = tempfile.mkdtemp()
    
    def create_vector_store(self, documents: list):
        """Create vector store from documents"""
        if not documents:
            raise ValueError("No documents to process")
        
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        return len(documents)
    
    def get_retriever(self):
        """Get retriever from vector store"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        return self.vector_store.as_retriever(
            search_kwargs={'k': 3}
        )
    
    def cleanup(self):
        """Clean up temporary files"""
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
