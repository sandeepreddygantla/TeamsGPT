# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Meetings AI is a Flask-based document analysis and chat application that processes meeting documents using OpenAI/Azure OpenAI LLM technologies with modular architecture for AI-powered semantic search and conversations.

## Key Commands

```bash
# Run application
python flask_app.py
# Visit: http://127.0.0.1:5000/meetingsai/

# Install dependencies
pip install -r requirements.txt

# Create .env file with:
OPENAI_API_KEY=your_api_key                 # For OpenAI
SECRET_KEY=your-secure-random-key           # Required for Flask sessions
BASE_PATH=/meetingsai                       # Optional, defaults to /meetingsai

# OR for Azure (see azure_meeting_processor_reference.py):
# AZURE_CLIENT_ID=your_azure_client_id
# AZURE_CLIENT_SECRET=your_azure_client_secret  
# AZURE_PROJECT_ID=your_azure_project_id

# User management
python user_manager.py      # Add/manage users
python user_removal.py      # Remove users
```

## Critical Architecture Patterns

### Global AI Variables Pattern (MANDATORY)
**NEVER instantiate AI clients directly.** Always use globals from `meeting_processor.py`:

```python
from meeting_processor import access_token, embedding_model, llm

# Always check for None before using
if llm is not None:
    response = llm.invoke(prompt)
else:
    logger.error("LLM not available - check API key configuration")
```

**NEVER DO:** `ChatOpenAI()`, `OpenAIEmbeddings()`, `AzureChatOpenAI()`

### Service Composition Pattern
All services share a single DatabaseManager instance:
```python
db_manager = DatabaseManager()
services = {
    'auth': AuthService(db_manager),
    'chat': ChatService(db_manager, processor),
    'document': DocumentService(db_manager),
    'upload': UploadService(db_manager, processor)
}
```

### Environment Switching (OpenAI ↔ Azure)
Modify only these functions in `meeting_processor.py`:
- `get_access_token()` - Return None for OpenAI, Azure token for Azure
- `get_llm()` - Return ChatOpenAI or AzureChatOpenAI with model selection
- `get_embedding_model()` - Return OpenAIEmbeddings or AzureOpenAIEmbeddings

## Core Components

- **`flask_app.py`** - Main Flask application with blueprints
- **`meeting_processor.py`** - Global AI variables: `access_token`, `embedding_model`, `llm`, model switching
- **`src/database/`** - DatabaseManager (SQLite + FAISS hybrid)
- **`src/services/`** - Business logic services  
- **`src/api/`** - Flask blueprints for routes
- **`src/ai/`** - LLM operations and enhanced query processing
- **`azure_meeting_processor_reference.py`** - Azure deployment variant with GPT-5/GPT-4.1

## Database Architecture

**Hybrid Storage:**
- `meeting_documents.db` - SQLite for metadata, users, projects
- `vector_index.faiss` - FAISS for semantic embeddings (auto-rebuilds if deleted)
- `sessions.db` - Session storage

**Access Pattern:**
```python
db_manager = DatabaseManager()
documents = db_manager.get_all_documents(user_id)
```

## Model Selection System

Dynamic LLM switching with organization models:
- **Models:** GPT-5, GPT-4.1 in `AVAILABLE_MODELS` dict
- **API:** `/api/model/available`, `/api/model/current`, `/api/model/switch`
- **Functions:** `set_current_model()`, `get_current_model_name()`, `get_current_model_config()`
- **Persistence:** localStorage saves user preference

## Processing Pipeline

1. **Upload:** File validation → SHA-256 deduplication → Content extraction
2. **AI Analysis:** LLM metadata extraction (topics, participants, decisions)
3. **Chunking:** RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
4. **Embedding:** text-embedding-3-large → FAISS storage
5. **Background:** ThreadPoolExecutor with job tracking

## Frontend Architecture

- **Main:** `static/script.js` - Core application logic
- **Config:** `static/js/config.js` - Base path management
- **Mentions:** `static/js/modules/mentions.js` - Advanced filtering (@project, @meeting, @date, #folder)
- **Model UI:** Real-time model switching with visual feedback

## IIS Deployment

**web.config requirements:**
- WSGI Handler: `flask_app.app`
- Base Path: `/meetingsai` via `BASE_PATH` env var
- FastCGI for Windows IIS deployment

## Key Features

### Default Project Upload Confirmation
- Modal for "Default Project" or empty selection
- "Continue Upload" button color: `#FF612B`
- ESC/outside click to close

### Enhanced Search Processing
ChatService uses enhanced processing for:
- Queries expecting 10+ documents
- Complex multi-meeting summaries
- Date range queries
- Project-wide analysis

## Performance Optimizations

- **Vector Batch Processing:** 100 vectors/batch for memory efficiency
- **Tiktoken Cache:** Directory at `tiktoken_cache/`
- **SQLite WAL Mode:** Connection pooling
- **Session Backend:** Custom SQLite for IIS compatibility

## Troubleshooting

```bash
# Vector database sync issues
rm vector_index.faiss  # Forces rebuild on restart

# Check environment setup
python -c "
from meeting_processor import llm, embedding_model, get_current_model_name
print(f'LLM: {\"✓\" if llm else \"✗\"}')
print(f'Embedding: {\"✓\" if embedding_model else \"✗\"}')
print(f'Current Model: {get_current_model_name()}')"

# Database inspection
sqlite3 meeting_documents.db "SELECT COUNT(*) FROM documents;"
python -c "import faiss; print(f'Vectors: {faiss.read_index(\"vector_index.faiss\").ntotal}')"
```

## Logs
- `logs/flask_app.log` - Application logs
- `logs/meeting_processor.log` - LLM initialization logs

## Additional Documentation
- **Azure Deployment:** See `AZURE_DEPLOYMENT_GUIDE.md` for Azure OpenAI setup
- **Architecture:** See `AZURE_ARCHITECTURE_OVERVIEW.md` for cloud migration plans
- **Token Optimization:** See `MEETINGS_AI_TOKEN_OPTIMIZATION_REPORT.md` for efficiency analysis