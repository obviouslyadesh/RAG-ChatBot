from app.rag import ask

while True:
    q = input("\nAsk: ")

    if q.lower() in ["exit", "quit"]:
        break

    answer, docs = ask(q)

    print("\nAnswer:\n", answer)

    print("\nSources:")
    for d in docs:
        print("-", d.metadata)