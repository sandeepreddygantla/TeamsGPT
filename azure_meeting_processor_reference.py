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
current_model_name = "gpt-5"  # Default model

# Available Azure models configuration for your organization
AVAILABLE_MODELS = {
    "gpt-5": {
        "name": "GPT-5",
        "azure_deployment": "gpt-5_2025-08-07",
        "model": "gpt-5",
        "api_version": "2025-01-01-preview",
        "temperature": 0,
        "max_tokens": 16000,
        "description": "Most advanced model for complex reasoning"
    },
    "gpt-4.1": {
        "name": "GPT-4.1", 
        "azure_deployment": "gpt-4.1_2025-04-14",
        "model": "gpt-4.1",
        "api_version": "2025-01-01-preview",
        "temperature": 0,
        "max_tokens": 16000,
        "description": "Enhanced GPT-4 model for balanced performance"
    }
}

def get_current_model_config():
    """Get the current model configuration"""
    return AVAILABLE_MODELS.get(current_model_name, AVAILABLE_MODELS["gpt-5"])

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
        model_config = AVAILABLE_MODELS.get(model_name or current_model_name, AVAILABLE_MODELS["gpt-5"])
        
        return AzureChatOpenAI(
            azure_deployment=model_config["azure_deployment"],
            model=model_config["model"],
            api_version=model_config["api_version"],
            azure_endpoint="https://api.uhg.com/api/cloud/api-management/ai-gateway/1.0",
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

# =====================================
# DEPLOYMENT INSTRUCTIONS FOR ORGANIZATION
# =====================================

# Step 1: Replace Functions in meeting_processor.py
# Replace these exact functions with the Azure versions above:
# - get_access_token()
# - get_llm()
# - get_embedding_model()
# - AVAILABLE_MODELS dictionary
# - current_model_name default value

# Step 2: Environment Variables
# Set these environment variables in your organization:
# AZURE_CLIENT_ID=your_azure_client_id
# AZURE_CLIENT_SECRET=your_azure_client_secret  
# AZURE_PROJECT_ID=your_azure_project_id

# Step 3: Verify Azure Deployment Names
# Ensure these deployment names match your Azure setup:
# - gpt-5_2025-08-07 (for GPT-5)
# - gpt-4.1_2025-04-14 (for GPT-4.1)
# - text-embedding-3-large_1 (for embeddings)

# Step 4: Update Frontend Default (Optional)
# In templates/chat.html, line 76:
# Change: <span class="model-name" id="current-model-name">GPT-5</span>

# Step 5: Test the Deployment
# 1. Start the application
# 2. Open browser console and run: testModelFeatures()
# 3. Verify both models appear in dropdown
# 4. Test switching between GPT-5 and GPT-4.1
# 5. Confirm chat responses show correct model names

# Note: All model switching, persistence, and confirmation features will work
# identically to the OpenAI version with these Azure models.