from fastapi import FastAPI, UploadFile, File
import shutil
import os

from app.core.rag import generate_answer,retriever
from app.services.ingest import ingest_pdf

app = FastAPI(title="RAG SaaS API")

UPLOAD_DIR = "app/data/documents"


@app.post("/chat")
def chat(payload: dict):
    return generate_answer(payload["question"])


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    path = f"app/data/documents/{file.filename}"

    with open(path, "wb") as f:
        f.write(file.file.read())

    chunks_created = ingest_pdf(path)

    return {
        "message": "file processed",
        "chunks_created": chunks_created
    }