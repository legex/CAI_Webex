from apigateway.services.rag_engine import RagEngine
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
rg = RagEngine()

class Query(BaseModel):
    query: str

@app.get("/")
def root():
    return {"message": "This is API router for Chatbot"}

@app.post("/invoke")
def invoke_model(request: Query):
    question = request.query
    response = rg.generate_response(question)
    return response
