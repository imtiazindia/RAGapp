# rag_engine.py
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever

from config import OPENAI_CONFIG

class RAGEngine:
    """RAG query processing engine"""
    
    def __init__(self, retriever: BaseRetriever):
        self.llm = ChatOpenAI(
            model_name=OPENAI_CONFIG['chat_model'],
            temperature=OPENAI_CONFIG['temperature'],
            openai_api_key=OPENAI_CONFIG['api_key'],
            streaming=True
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
    
    def query(self, question: str):
        """Query the RAG system"""
        result = self.qa_chain({"query": question})
        return result["result"], result["source_documents"]
