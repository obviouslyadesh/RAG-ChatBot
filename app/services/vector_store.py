from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def load_vectorstore():
    return FAISS.load_local(
        "app/data/vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )