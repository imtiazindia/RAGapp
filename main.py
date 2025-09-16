# main.py
import streamlit as st
import os

from config import SUPPORTED_EXTENSIONS, OPENAI_CONFIG
from document_processor import DocumentProcessor
from embedding_manager import EmbeddingManager
from rag_engine import RAGEngine
from ui_components import setup_page, render_header, render_sidebar, render_main_content, render_answer, render_error

def initialize_session_state():
    """Initialize session state variables"""
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = 0
    if "supported_extensions" not in st.session_state:
        st.session_state.supported_extensions = list(SUPPORTED_EXTENSIONS.keys())
    if "api_key" not in st.session_state:
        st.session_state.api_key = OPENAI_CONFIG.get('api_key', '')

def main():
    # Setup page and initialize session state
    setup_page()
    initialize_session_state()
    render_header()
    
    # Render sidebar and get inputs
    api_key, uploaded_files, process_btn = render_sidebar()
    
    # Update API key in session state and environment
    if api_key:
        st.session_state.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Process documents if button is clicked
    if uploaded_files and process_btn:
        if not st.session_state.api_key:
            render_error("Please enter your OpenAI API key in the sidebar.")
            return
        
        with st.spinner("Processing documents..."):
            try:
                # Initialize document processor
                processor = DocumentProcessor()
                
                # Load and split documents
                documents = processor.load_documents(uploaded_files)
                chunks = processor.split_documents(documents)
                
                # Create embeddings and vector store
                embedding_manager = EmbeddingManager()
                num_chunks = embedding_manager.create_vector_store(chunks)
                
                # Store retriever in session state
                st.session_state.retriever = embedding_manager.get_retriever()
                st.session_state.processed = True
                st.session_state.documents_processed = num_chunks
                
                st.success(f"Processed {len(documents)} documents into {num_chunks} chunks!")
            except Exception as e:
                render_error(f"Error processing documents: {str(e)}")
    
    # Render main content and handle queries
    question = render_main_content(
        st.session_state.processed, 
        st.session_state.documents_processed
    )
    
    if question and st.session_state.processed:
        with st.spinner("Thinking..."):
            try:
                # Initialize RAG engine
                rag_engine = RAGEngine(st.session_state.retriever)
                
                # Process query
                answer, sources = rag_engine.query(question)
                
                # Render results
                render_answer(answer, sources)
            except Exception as e:
                render_error(f"Error querying documents: {str(e)}")

if __name__ == "__main__":
    main()
