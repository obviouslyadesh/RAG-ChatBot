from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

VECTOR_DB_PATH = "app/data/vectorstore"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def load_vectorstore():
    return FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOllama(
    model="llama3",
    temperature=0
)


def reload_vectorstore():
    global vectorstore, retriever

    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


def generate_answer(question: str):
    docs = retriever.invoke(question)

    context = "\n\n".join(
        d.page_content for d in docs
    )

    prompt = f"""
    You are a RAG assistant.

    Rules:
    1. Answer ONLY from the provided context.
    2. If the answer is not found in context, say:
    "I don't know based on the provided documents."
    3. Do NOT contradict yourself.
    4. If the user requests Nepali, answer in Nepali.
    5. Be concise and factual.

    Context:
    {context}

    Question:
    {question}
    """

    answer = llm.invoke(prompt).content

    return {
        "answer": answer,
        "sources": [
            {
                "file": d.metadata.get("source"),
                "page": d.metadata.get("page")
            }
            for d in docs
        ]
    }