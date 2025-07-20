from apigateway.services.rag_engine import RagEngine
from webexteamssdk import WebexTeamsAPI
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel

load_dotenv(r'datamanagement\core\.env')
app = FastAPI()
rg = RagEngine()
api = WebexTeamsAPI(access_token=os.getenv("WEBEX_TOKEN"))

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

@app.post("/webexhook")
async def webhook(request: Request):
    payload = await request.json()
    data = payload.get("data", {})
    message_id = data.get("id")
    room_id = data.get("roomId")
    
    message = api.messages.get(message_id)
    user_email = message.personEmail
    user_text = message.text

    if user_email == "localhelper@webex.bot":
        return {"message": "Ignoring bot's own message"}

    response = rg.generate_response(user_text)
    api.messages.create(roomId=room_id, text=response)
    return {"message": "Response sent"}
