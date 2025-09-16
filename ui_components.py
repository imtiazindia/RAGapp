# ui_components.py
import streamlit as st

def setup_page():
    """Setup Streamlit page configuration"""
    st.set_page_config(
        page_title="RAG Document Assistant",
        page_icon="📄",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
    }
    .success-box {
        padding: 15px;
        border-radius: 5px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    """Render application header"""
    st.markdown('<h1 class="main-header">📄 RAG Document Assistant</h1>', unsafe_allow_html=True)
    st.markdown("Upload your documents and ask questions about their content.")

def render_sidebar():
    """Render sidebar components"""
    with st.sidebar:
        st.markdown('<h2 class="sub-header">Configuration</h2>', unsafe_allow_html=True)
        
        # API key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.get("api_key", ""),
            help="Enter your OpenAI API key."
        )
        
        # File upload
        st.markdown("### Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=list(st.session_state.get("supported_extensions", [])),
            accept_multiple_files=True
        )
        
        process_btn = st.button("Process Documents")
    
    return api_key, uploaded_files, process_btn

def render_main_content(processed, documents_processed=0):
    """Render main content based on application state"""
    if processed:
        st.markdown("### Ask Questions About Your Documents")
        
        question = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know about your documents?"
        )
        
        if documents_processed > 0:
            st.markdown(f'<div class="success-box">Processed {documents_processed} document chunks</div>', 
                       unsafe_allow_html=True)
        
        return question
    else:
        st.info("Please upload and process documents to start asking questions.")
        return None

def render_answer(answer, sources):
    """Render answer and sources"""
    if answer:
        st.markdown("### Answer")
        st.write(answer)
        
        if sources:
            st.markdown("### Source Documents")
            for i, source in enumerate(sources):
                with st.expander(f"Source {i+1}: {source.metadata.get('source', 'Unknown')}"):
                    st.write(source.page_content)

def render_error(message):
    """Render error message"""
    st.error(message)
