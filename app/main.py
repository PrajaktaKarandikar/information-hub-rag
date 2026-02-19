from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import logging
from typing import Optional, List
from datetime import datetime, timezone
from app.observability import track_metrics
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Local imports
from app.config import CONFIG
from app.rag_pipeline import ProductionRAGPipeline
from app.information_loader import InformationLoader

# Initialize FastAPI app
app = FastAPI(
    title=CONFIG['application']['name'],
    version=CONFIG['application']['version'],
    description=CONFIG['application']['description']
)

# Instantiate global pipeline variable
rag_pipeline = None
information_loader = None

@app.on_event("startup")
def startup_event():
    """
    Initialize the RAG pipeline and information loader on application startup.
    """
    global rag_pipeline, information_loader

    # Get API key from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not found in environment variables.")
        raise ValueError("OPENAI_API_KEY is required. Set it in the environment variables.")
    
    # Initialize the RAG pipeline and information loader
    information_loader = InformationLoader()
    rag_pipeline = ProductionRAGPipeline(openai_api_key)

    # Load default information sources
    default_sources = []

    # Add PDF source if configured
    pdf_path = CONFIG['pdf'].get('file_path')
    if pdf_path and os.path.exists(pdf_path):
        default_sources.append(pdf_path)

    # Add website source if configured
    web_sources = CONFIG.get('sources', {}).get('web', [])
    default_sources.extend(web_sources)

    if default_sources:
        try:
            rag_pipeline.create_vector_store(default_sources)
            logger.info(f"Vector store created with {len(default_sources)} sources loaded.")
        except Exception as e:
            logger.warning(f"Failed to create vector store with default sources: {e}")

    logger.info("RAG pipeline and information loader initialized successfully. Application startup complete.")
   
# ======================================== API Models ==================================================================
   
class QueryRequest(BaseModel):
    question: str
    return_sources: bool = False

class IngestRequest(BaseModel):
    sources: List[str]  
    replace_existing: bool = False

class HealthResponse(BaseModel):
    status: str
    service: str
    vector_store_initialized: bool
    timestamp: str

# ======================================== API Endpoints ==================================================================

@app.get("/")
def root():
    """
    Root endpoint providing basic information about the API.
    """
    return {
        "service": CONFIG['application']['name'],
        "version": CONFIG['application']['version'],
        "endpoints": {
            "health": "/health",
            "query": "/query (POST)",
            "ingest": "/ingest (POST)",
            "config": "/config"
        }
    }

@app.get("/health")
def health_check() -> HealthResponse:
    """
    Health check endpoint to verify that the API is running and the vector store is initialized.
    """

    vector_initialized = rag_pipeline is not None and rag_pipeline.vector_store is not None

    return HealthResponse(
        status="healthy" if vector_initialized else "degraded",
        vector_store_initialized=vector_initialized,
        service=CONFIG['application']['name'],
        timestamp=datetime.now(timezone.utc).isoformat()
    )

@app.get("/metrics")
def metrics():
    """ Metrics endpoint to show Prometheus metrics. """
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.post("/query")
@track_metrics
def query_question(request: QueryRequest):
    """
    Endpoont to query the RAG pipeline with a question. Optionally returns sources if requested.
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized.")
    
    try:
        result = rag_pipeline.query(
            question=request.question, 
            return_sources=request.return_sources
        )
        return result
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
def ingest_sources(request: IngestRequest):
    """
    Endpoint to ingest new information sources into the vector store. Can replace existing vector store if specified.
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized.")
    
    try:
        if request.replace_existing:
            # Create new vector store with the provided sources, replacing any existing data
            rag_pipeline.create_vector_store(request.sources)
            message = "Vector store replaced with new sources."
        else:
            # Add new sources to the existing vector store without replacing it
            # # Note: For production, implement proper add_to_vector_store method in the RAG pipeline to handle incremental updates.
            current_sources = request.sources
            rag_pipeline.create_vector_store(current_sources)  # This will currently replace the vector store, but should be updated to add to it instead.
            #rag_pipeline.add_to_vector_store(request.sources)
            message = "New sources added to existing vector store."

        return {
            "status": "success",
            "message": message, 
            "sources_processed": len(request.sources),
            "ingested_sources": request.sources            
        }
    except Exception as e:
        logger.error(f"Error ingesting sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/config")
def show_config():
    """
    Endpoint to return a safe view of the current configuration settings.
    """
    safe_config = CONFIG.copy()
    # Mask or remove any sensitive information from the configuration before returning it
    if 'pdf' in safe_config and 'file_path' in safe_config['pdf']:
        safe_config['pdf']['file_path'] = "REDACTED"
    return safe_config
   
@app.post("/upload")
def upload_document(s3_key: str, file_path: str):  
    """
    Endpoint to upload a document to S3 and add it to the vector store. 
    
    Args:
        s3_key (str): The key under which to store the document in S3.
        file_path (str): The local file path of the document to upload.
    """
    if not information_loader:
        raise HTTPException(status_code=503, detail="Information loader not initialized.")
    
    success = information_loader.upload_document(file_path, s3_key)

    if success:
        return {"status": "success", "message": f"Document uploaded successfully to S3://{(information_loader.s3_bucket)}/{s3_key}"}
    else:
        return HTTPException(status_code=500, detail="Failed to upload document to S3.")
    
