from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DB_DIR = "app/data/vectorstore"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def ingest_pdf(file_path: str):
    """
    Ingest a PDF into the FAISS vector database.

    If an index already exists:
        -> append new chunks

    Otherwise:
        -> create a new vector database
    """

    # ---------------------------
    # Load PDF
    # ---------------------------

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # ---------------------------
    # Split into chunks
    # ---------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)

    # ---------------------------
    # Check existing DB
    # ---------------------------

    db_exists = (
        Path(DB_DIR, "index.faiss").exists()
        and
        Path(DB_DIR, "index.pkl").exists()
    )

    # ---------------------------
    # Update existing DB
    # ---------------------------

    if db_exists:

        vectorstore = FAISS.load_local(
            DB_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )

        vectorstore.add_documents(chunks)

        print(
            f"✅ Added {len(chunks)} chunks to existing vector DB"
        )

    # ---------------------------
    # Create new DB
    # ---------------------------

    else:

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

        print(
            f"✅ Created new vector DB with {len(chunks)} chunks"
        )

    # ---------------------------
    # Save DB
    # ---------------------------

    vectorstore.save_local(DB_DIR)

    print("✅ Vector DB saved")

    return len(chunks)