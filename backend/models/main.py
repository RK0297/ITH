from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

# Import our modules (assuming they're in the same package)
from vector_database import VectorDatabase, load_qa_data
try:
    from rag_pipeline import RAGPipeline
except Exception:
    RAGPipeline = None

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Legal Chatbot API",
    description="API for querying Indian Constitution and Legal Cases",
    version="1.0.0"
)

# Configure CORS - Allow all origins for development
# In production, replace ["*"] with specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Set to False when using allow_origins=["*"]
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Initialize components
vector_database = None
rag_pipeline = None

# Pydantic models
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    top_k: Optional[int] = Field(5, ge=1, le=10, description="Number of relevant documents to retrieve")

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict]
    conversation_id: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    vector_database_status: str
    vector_database_count: int
    ollama_status: str
    timestamp: str

class StatsResponse(BaseModel):
    total_documents: int
    categories: List[str]
    document_types: List[str]
    collection_name: str


class BuildRequest(BaseModel):
    reset: Optional[bool] = Field(False, description="If true, reset the DB before adding data")
    max_items: Optional[int] = Field(None, description="Max number of items to add (for testing)")
    filename: Optional[str] = Field("legal_data_all.json", description="Data filename to load from data/raw")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global vector_database, rag_pipeline
    
    logger.info("Initializing AI Legal Chatbot API...")
    
    try:
        # Initialize vector database
        vector_database = VectorDatabase(
            persist_directory=os.getenv("VECTOR_DB_PATH", "../data/vectordb"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            collection_name=os.getenv("VECTOR_DB_COLLECTION", "legal_qa")
        )
        logger.info("Vector database initialized")
        
        # Initialize RAG pipeline
        if RAGPipeline:
            try:
                rag_pipeline = RAGPipeline(
                    vector_database=vector_database,
                    ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
                    ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                )
                logger.info("RAG pipeline initialized")
            except Exception as e:
                logger.warning(f"RAG pipeline could not be initialized: {e}")
                rag_pipeline = None
        else:
            logger.info("RAG pipeline not available (module import failed). Continuing without it.")
        
        logger.info("API startup complete")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

# Routes
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "AI Legal Chatbot API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and component status"""
    try:
        # Check vector DB
        vector_db_count = vector_database.collection.count() if vector_database else 0
        vector_db_status = "healthy" if vector_db_count > 0 else "empty"
        
        # Check Ollama
        ollama_status = "healthy" if rag_pipeline and rag_pipeline.check_ollama() else "unavailable"
        
        return HealthResponse(
            status="healthy",
            vector_db_status=vector_db_status,
            vector_db_count=vector_db_count,
            ollama_status=ollama_status,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats", response_model=StatsResponse, tags=["Statistics"])
async def get_stats():
    """Get database statistics"""
    try:
        if not vector_database:
            raise HTTPException(status_code=500, detail="Vector database not initialized")
        
        stats = vector_database.get_stats()
        return StatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.options("/api/chat", tags=["Chat"])
async def chat_options():
    """Handle CORS preflight for chat endpoint"""
    return {"message": "OK"}

@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """Process chat query and return response"""
    try:
        if not rag_pipeline:
            raise HTTPException(status_code=500, detail="RAG pipeline not initialized")
        
        logger.info(f"Processing query: {request.query[:100]}...")
        
        # Generate response using RAG pipeline
        result = rag_pipeline.query(
            query=request.query,
            top_k=request.top_k,
            conversation_id=request.conversation_id
        )
        
        # Prepare sources
        sources = []
        for doc, metadata in zip(result['documents'], result['metadatas']):
            source = {
                "title": metadata.get('title', metadata.get('case_name', 'Unknown')),
                "source": metadata.get('source', 'Unknown'),
                "category": metadata.get('category', 'Unknown'),
                "url": metadata.get('url', ''),
                "preview": doc[:200] + "..." if len(doc) > 200 else doc
            }
            sources.append(source)
        
        return ChatResponse(
            response=result['response'],
            sources=sources,
            conversation_id=result['conversation_id'],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search", tags=["Search"])
async def search_documents(query: str, top_k: int = 5):
    """Search for relevant documents without generating response"""
    try:
        if not vector_database:
            raise HTTPException(status_code=500, detail="Vector database not initialized")
        
        results = vector_database.search(query, n_results=top_k)

        documents = []
        docs = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0] if 'distances' in results else [None] * len(docs)

        for doc, metadata, dist in zip(docs, metadatas, distances):
            documents.append({
                "id": metadata.get('id'),
                "instruction": metadata.get('instruction'),
                "response": metadata.get('response'),
                "content_preview": doc[:400] + "..." if len(doc) > 400 else doc,
                "score": None if dist is None else float(dist)
            })

        return {
            "query": query,
            "results": documents,
            "count": len(documents)
        }
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/build-db", tags=["Admin"])
async def build_database(req: BuildRequest):
    """Build or update the vector database from the JSON files.

    This endpoint allows the frontend or admin to trigger a non-interactive build.
    """
    try:
        if not vector_database:
            raise HTTPException(status_code=500, detail="Vector database not initialized")

        if req.reset:
            vector_database.reset_database()

        data = load_qa_data(req.filename, "../scrapers/data/raw")
        if not data:
            raise HTTPException(status_code=404, detail="No data found to load")

        if req.max_items:
            data = data[:req.max_items]

        vector_database.add_documents(data, batch_size=100)

        return {"status": "ok", "added": len(data)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error building DB: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/doc/{doc_id}", tags=["Documents"])
async def get_document(doc_id: str):
    """Fetch a document by its numeric id or 'qa_<id>' format."""
    try:
        if not vector_database:
            raise HTTPException(status_code=500, detail="Vector database not initialized")

        # Normalize id
        if doc_id.startswith("qa_"):
            lookup_id = doc_id
        else:
            lookup_id = f"qa_{doc_id}"

        result = vector_database.collection.get(ids=[lookup_id])

        if not result or not result.get('ids'):
            raise HTTPException(status_code=404, detail="Document not found")

        # Chroma returns lists per field
        return {
            "id": result['ids'][0],
            "document": result.get('documents', [[]])[0][0] if result.get('documents') else None,
            "metadata": result.get('metadatas', [[]])[0][0] if result.get('metadatas') else None
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn main:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)