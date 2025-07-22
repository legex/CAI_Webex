"""
Module: api_router.py

Sets up the FastAPI router for an AI-powered chatbot API with Webex integration.
Implements request logging, error logging, and key workflow logging for both standard and webhook endpoints.
Integrates with RagEngine for Retrieval-Augmented Generation and sends responses to Webex Teams.

Logging configuration is managed using the setup_logger utility.
"""
from apigateway.services.rag_engine import RagEngine
from webexteamssdk import WebexTeamsAPI
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel
from datamanagement.core.logger import setup_logger

logger = setup_logger('api_router', 'datamanagement/log/api_router.log')

load_dotenv(r'datamanagement\core\.env')
app = FastAPI()
rg = RagEngine()
api = WebexTeamsAPI(access_token=os.getenv("WEBEX_TOKEN"))

class Query(BaseModel):
    query: str

@app.get("/")
def root():
    logger.info("Root endpoint called.")
    return {"message": "This is API router for Chatbot"}

@app.post("/invoke")
def invoke_model(request: Query):
    question = request.query
    logger.info(f"POST /invoke received with query: {question}")
    try:
        response = rg.generate_response(question)
        logger.info("Response generated successfully for /invoke endpoint")
        return response
    except Exception as e:
        logger.error(f"Error during /invoke handling: {e}", exc_info=True)
        return {"error": "Failed to generate response."}

@app.post("/webexhook")
async def webhook(request: Request):
    try:
        payload = await request.json()
        logger.info(f"Webhook triggered with payload: {payload}")
        data = payload.get("data", {})
        message_id = data.get("id")
        room_id = data.get("roomId")
        
        message = api.messages.get(message_id)
        user_email = message.personEmail
        user_text = message.text

        logger.info(f"Webhook message from: {user_email}, text: {user_text}")
        if user_email == "localhelper@webex.bot":
            logger.info("Ignoring bot's own message (loop prevention).")
            return {"message": "Ignoring bot's own message"}

        response = rg.generate_response(user_text)
        logger.info("Responded to Webex message in room %s.", room_id)
        api.messages.create(roomId=room_id, text=response)
        return {"message": "Response sent"}

    except Exception as e:
        logger.error(f"Error during Webex webhook processing: {e}", exc_info=True)
        return {"error": "Webhook processing failed."}
