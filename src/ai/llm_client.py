"""
LLM client module for Meetings AI application.
Follows instructions.md requirements - uses global variables from meeting_processor.
"""
import logging
from typing import Optional, Dict, Any, List

# Import global variables as per instructions.md requirements
from meeting_processor import access_token, embedding_model, llm

logger = logging.getLogger(__name__)


def initialize_ai_clients():
    """
    Initialize global AI client instances following instructions.md.
    
    NOTE: This function is a compatibility wrapper.
    The actual initialization happens in meeting_processor.py using:
    - access_token = get_access_token()
    - embedding_model = get_embedding_model(access_token)  
    - llm = get_llm(access_token)
    
    Returns:
        bool: True if clients are available, False otherwise
    """
    try:
        logger.info("Initializing AI clients...")
        
        # Check if global variables are properly initialized
        if embedding_model is None or llm is None:
            logger.error("Global AI clients not initialized in meeting_processor")
            return False
            
        logger.info("Embedding model initialized successfully")
        logger.info("LLM initialized successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to access AI clients: {e}")
        return False


