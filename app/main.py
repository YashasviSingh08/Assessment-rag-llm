from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.rag import ask
from app.embeddings import build_vector_store
from app.logger import logger

app = FastAPI(
    title="Healthcare AI Assistant",
    description="Medical RAG Assistant using MedQuAD, FAISS, Sentence Transformers, and Ollama",
    version="1.0.0"
)


# =====================================================
# Request Model
# =====================================================

class QuestionRequest(BaseModel):
    question: str


# =====================================================
# Home
# =====================================================

@app.get("/")
def home():

    return {
        "message": "Healthcare AI Assistant API is running.",
        "docs": "/docs",
        "health": "/health"
    }


# =====================================================
# Health Check
# =====================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "Healthcare AI Assistant",
        "version": "1.0.0"
    }


# =====================================================
# Ask Question
# =====================================================

@app.post("/ask")
def ask_question(request: QuestionRequest):

    try:

        result = ask(request.question)

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Error while answering question: {str(e)}"
        )


# =====================================================
# Rebuild Vector Store
# =====================================================

@app.post("/ingest")
def ingest():

    try:

        result = build_vector_store()

        return {
            "status": "success",
            "message": "Vector store rebuilt successfully.",
            "documents_indexed": result["documents"],
            "vectors": result["vectors"],
            "index_file": result["index"],
            "metadata_file": result["metadata"]
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Error during ingestion: {str(e)}"
        )


# =====================================================
# Run Server
# =====================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )