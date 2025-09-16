# config.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    '.pdf': 'PDF',
    '.docx': 'Word',
    '.pptx': 'PowerPoint',
    '.xlsx': 'Excel'
}

# OpenAI configuration
OPENAI_CONFIG = {
    'embedding_model': 'text-embedding-ada-002',
    'chat_model': 'gpt-3.5-turbo',
    'temperature': 0,
    'api_key': os.getenv('OPENAI_API_KEY')
}

# Text splitting configuration
TEXT_SPLITTER_CONFIG = {
    'chunk_size': 1000,
    'chunk_overlap': 200
}

# Vector store configuration
VECTOR_STORE_CONFIG = {
    'search_kwargs': {'k': 3}
}
