from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

query = input("Ask a question: ")

results = vectorstore.similarity_search(
    query,
    k=3
)

print("\nTop Results\n")

for i, doc in enumerate(results, start=1):
    print(f"\n===== Result {i} =====")
    print(doc.page_content[:500])

    print("\nMetadata:")
    print(doc.metadata)