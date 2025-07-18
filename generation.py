from app.services.rag_engine import RagEngine

rg = RagEngine()
while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break

    response = rg.generate_response(question)
    print(response)
