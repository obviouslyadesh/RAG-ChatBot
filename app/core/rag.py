import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

VECTOR_DB_PATH = "app/data/vectorstore"

# --------------------------------------------------
# Embedding Model
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------------------------------
# LLM (Groq)
# --------------------------------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# --------------------------------------------------
# Vector Store
# --------------------------------------------------

vectorstore = None
retriever = None


def load_vectorstore():
    """
    Load FAISS index if available.
    """

    faiss_file = os.path.join(
        VECTOR_DB_PATH,
        "index.faiss"
    )

    if not os.path.exists(faiss_file):
        print("⚠️ No vector database found.")
        return None

    print("✅ Loading FAISS index...")

    return FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


def initialize_retriever():
    """
    Initialize retriever at startup.
    """

    global vectorstore, retriever

    vectorstore = load_vectorstore()

    if vectorstore:
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
        print("✅ Retriever ready")
    else:
        retriever = None
        print("⚠️ Retriever not initialized")


initialize_retriever()

# --------------------------------------------------
# Reload after uploads
# --------------------------------------------------


def reload_vectorstore():
    """
    Reload vector DB after a new PDF upload.
    """

    print("🔄 Reloading vectorstore...")
    initialize_retriever()


# --------------------------------------------------
# Main RAG Function
# --------------------------------------------------


def generate_answer(question: str):

    if retriever is None:
        return {
            "answer": (
                "No documents have been uploaded yet. "
                "Please upload a PDF first."
            ),
            "sources": []
        }

    docs = retriever.invoke(question)

    context = "\n\n".join(
        d.page_content
        for d in docs
    )

    prompt = f"""
You are a professional RAG assistant.

RULES:
1. Use ONLY the provided context.
2. If answer is not found, say:
   "I don't know based on the provided documents."
3. Do not invent facts.
4. If user asks in Nepali, answer in Nepali.
5. Be concise and accurate.

CONTEXT:
{context}

QUESTION:
{question}
"""

    try:
        answer = llm.invoke(prompt).content

    except Exception as e:
        return {
            "answer": f"LLM Error: {str(e)}",
            "sources": []
        }

    return {
        "answer": answer,
        "sources": [
            {
                "file": os.path.basename(
                    d.metadata.get("source", "")
                ),
                "page": d.metadata.get("page", 0) + 1
            }
            for d in docs
        ]
    }