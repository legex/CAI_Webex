import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from datamanagement.core.querychunking import ChunkEmbedRank as ce
from datamanagement.db.vector_query import VectorSearch


load_dotenv(dotenv_path=r'datamanagement\core\.env')
uri = os.getenv("MONGO_URI")
_model1 = OllamaLLM(model="llama3.2:1b", temperature=0)

template = """
    You are an expert technical assistant specialized in Cisco Collaboration services, including Webex, CUCM, Expressway, and CUBE/SBCs.

    You must answer ONLY using the information from the technical documents provided below. If you cannot find a relevant answer in the documents, clearly state: "The documents do not contain enough information to answer the query."

    ---
    Technical Documents:
    {technical_docs}
    ---

    Query:
    {question}

    Instructions:
    - Do not include information not supported by the documents.
    - If steps or commands are mentioned, ensure they are found in the documents.
    - If the product mentioned in the query is unclear, try to infer from context but do not hallucinate.
    - Keep the answer precise, technical, and structured if possible (e.g., steps, command blocks).
   """

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | _model1

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    
    vec_search =  VectorSearch(uri, database='cisco_docs',collection='dataset', query=question)
    search_response = vec_search.similarity_search()
    top_chunks = vec_search.rerank_results(search_response)
    technical_docs = []
    for response in top_chunks:
        technical_docs.append(response['response_chunk'])
    print(technical_docs)
    result = chain.invoke({"technical_docs":"\n\n".join(technical_docs), "question":question})
    print(result)
