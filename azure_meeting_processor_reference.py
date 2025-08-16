# Azure OpenAI Implementation Reference
# This file shows the Azure variant for your organization deployment
# Replace the corresponding functions in meeting_processor.py with these when deploying to Azure

import os
import httpx
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

# Global variable to store current model selection
current_model_name = "gpt-4o"  # Default model

# Available Azure models configuration
AVAILABLE_MODELS = {
    "gpt-4o": {
        "name": "GPT-4o",
        "azure_deployment": "gpt-4o_2024-05-13",
        "model": "gpt-4o",
        "api_version": "2025-01-01-preview",
        "temperature": 0.5,
        "max_tokens": 16000,
        "description": "Most capable model for complex reasoning"
    },
    "gpt-4o-mini": {
        "name": "GPT-4o Mini", 
        "azure_deployment": "gpt-4o-mini_2024-07-18",  # Adjust deployment name as per your Azure setup
        "model": "gpt-4o-mini",
        "api_version": "2025-01-01-preview",
        "temperature": 0.5,
        "max_tokens": 16000,
        "description": "Faster and more cost-effective model"
    }
}

def get_current_model_config():
    """Get the current model configuration"""
    return AVAILABLE_MODELS.get(current_model_name, AVAILABLE_MODELS["gpt-4o"])

def set_current_model(model_name: str):
    """Set the current model globally"""
    global current_model_name, llm
    if model_name in AVAILABLE_MODELS:
        current_model_name = model_name
        # Reinitialize LLM with new model
        try:
            access_token = get_access_token()
            llm = get_llm(access_token)
            logger.info(f"Successfully switched to Azure model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Error switching to Azure model {model_name}: {e}")
            return False
    else:
        logger.warning(f"Azure model {model_name} not available")
        return False

def get_current_model_name():
    """Get the current model name"""
    return current_model_name

# --- Azure Auth Function ---
def get_access_token():
    """Get Azure AD access token for authentication"""
    try:
        auth = "https://api.uhg.com/oauth2/token"
        scope = "https://api.uhg.com/.default"
        grant_type = "client_credentials"
        client_id = os.getenv("AZURE_CLIENT_ID")
        client_secret = os.getenv("AZURE_CLIENT_SECRET")
        
        if not client_id or not client_secret:
            logger.error("Azure credentials not found in environment variables")
            return None
            
        with httpx.Client() as client:
            body = {
                "grant_type": grant_type,
                "scope": scope,
                "client_id": client_id,
                "client_secret": client_secret
            }
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            response = client.post(auth, headers=headers, data=body, timeout=60)
            response.raise_for_status()
            return response.json()["access_token"]
    except Exception as e:
        logger.error(f"Error getting Azure access token: {e}")
        return None

# --- Azure OpenAI LLM Client ---
def get_llm(access_token: str, model_name: str = None):
    """
    Get Azure OpenAI LLM client with dynamic model selection.
    """
    try:
        if not access_token:
            logger.warning("Azure access token not available")
            return None
        
        # Use provided model_name or current global model
        model_config = AVAILABLE_MODELS.get(model_name or current_model_name, AVAILABLE_MODELS["gpt-4o"])
        
        return AzureChatOpenAI(
            azure_deployment=model_config["azure_deployment"],
            model=model_config["model"],
            api_version=model_config["api_version"],
            azure_endpoint="https://api.uhg.com/api/cloud/api-management/api-gateway/1.0",
            openai_api_type="azure_ad",
            validate_base_url=False,
            azure_ad_token=access_token,
            default_headers={"projectId": os.getenv("AZURE_PROJECT_ID")},
            temperature=model_config["temperature"],
            max_tokens=model_config["max_tokens"],
            request_timeout=120
        )
    except Exception as e:
        logger.error(f"Error creating Azure LLM client: {e}")
        return None

# --- Azure OpenAI Embedding Model ---
def get_embedding_model(access_token: str):
    """
    Get Azure OpenAI embedding model.
    Note: Embedding model is not changed based on user selection - only LLM models are switched.
    """
    try:
        if not access_token:
            logger.warning("Azure access token not available")
            return None
            
        return AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-3-large_1",
            api_version="2025-01-01-preview",
            azure_endpoint="https://api.uhg.com/api/cloud/api-management/ai-gateway/1.0",
            openai_api_type="azure_ad",
            validate_base_url=False,
            azure_ad_token=access_token,
            default_headers={
                "projectId": os.getenv("AZURE_PROJECT_ID"),
            },
        )
    except Exception as e:
        logger.error(f"Error creating Azure embedding model: {e}")
        return None

# Instructions for deployment:
# 1. Replace the corresponding functions in meeting_processor.py with these Azure versions
# 2. Ensure AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, and AZURE_PROJECT_ID are set in your environment
# 3. Update the AVAILABLE_MODELS dictionary to match your Azure deployment names
# 4. The model switching functionality will work the same way as the OpenAI version