from fastapi import FastAPI, UploadFile, File
import os

from app.core.rag import (
    generate_answer,
    reload_vectorstore
)

from app.services.ingest import ingest_pdf

app = FastAPI(title="RAG SaaS API")

UPLOAD_DIR = "app/data/documents"

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def root():
    return {
        "status": "healthy",
        "service": "RAG SaaS API"
    }


@app.post("/chat")
def chat(payload: dict):
    """
    Ask questions against uploaded documents.
    """

    question = payload.get("question")

    if not question:
        return {
            "error": "Question is required"
        }

    return generate_answer(question)


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    """
    Upload PDF -> Ingest -> Rebuild FAISS -> Reload Retriever
    """

    file_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    with open(file_path, "wb") as f:
        f.write(file.file.read())

    # Create / update vector DB
    chunks_created = ingest_pdf(file_path)

    # Reload retriever in memory
    reload_vectorstore()

    return {
        "message": "file processed successfully",
        "filename": file.filename,
        "chunks_created": chunks_created
    }