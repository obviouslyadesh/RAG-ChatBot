from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from app.core.rag import embeddings, reload_vectorstore

DB_DIR = "app/data/vectorstore"


def ingest_pdf(file_path):
    loader = PyPDFLoader(file_path)

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)

    db_exists = (
        Path(DB_DIR, "index.faiss").exists()
        and
        Path(DB_DIR, "index.pkl").exists()
    )

    if db_exists:

        vectorstore = FAISS.load_local(
            DB_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )

        vectorstore.add_documents(chunks)

    else:

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

    vectorstore.save_local(DB_DIR)

    reload_vectorstore()

    return len(chunks)