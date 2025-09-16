# embedding_manager.py
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

from config import OPENAI_CONFIG

class EmbeddingManager:
    """Manage embeddings and vector store"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=OPENAI_CONFIG['embedding_model'],
            openai_api_key=OPENAI_CONFIG['api_key']
        )
        self.vector_store = None
    
    def create_vector_store(self, documents: List[Document]):
        """Create vector store from documents"""
        if not documents:
            raise ValueError("No documents to process")
        
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        return len(documents)
    
    def get_retriever(self):
        """Get retriever from vector store"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        return self.vector_store.as_retriever(
            search_kwargs={'k': 3}
        )
