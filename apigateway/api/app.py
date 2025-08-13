"""
Module: api_router.py

Sets up the FastAPI router for an AI-powered chatbot API with Webex integration.
Implements request logging, error logging, and key workflow logging for both standard and webhook endpoints.
Integrates with RagEngine for Retrieval-Augmented Generation and sends responses to Webex Teams.

Logging configuration is managed using the setup_logger utility.
"""
import os
from pydantic import BaseModel
from webexteamssdk import WebexTeamsAPI
from dotenv import load_dotenv
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST
    )
from fastapi import FastAPI, Request
from fastapi.responses import Response
from datamanagement.core.logger import setup_logger
from apigateway.services.generate_response import get_response
from apigateway.api.utils import get_config_with_session

logger = setup_logger('api_router', 'datamanagement/log/api_router.log')

load_dotenv(r'datamanagement\core\.env')
app = FastAPI()
api = WebexTeamsAPI(access_token=os.getenv("WEBEX_TOKEN"))

# Prometheus metrics
REQUEST_COUNT = Counter(
    "api_requests_total", "Total API Requests", ["endpoint", "method"]
)
ERROR_COUNT = Counter(
    "api_errors_total", "Total API Errors", ["endpoint", "error_type"]
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds", "API Request latency in seconds", ["endpoint", "method"]
)

class Query(BaseModel):
    """
    Pydantic model for parsing the input query for the /invoke endpoint.
    """
    query: str

@app.get("/metrics")
def metrics():
    """
    Expose Prometheus metrics.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/")
def root():
    """
    Root endpoint for health check or basic info.
    """
    REQUEST_COUNT.labels(endpoint="/", method="GET").inc()
    with REQUEST_LATENCY.labels(endpoint="/", method="GET").time():
        logger.info("Root endpoint called.")
        return {"message": "This is API router for Chatbot"}

@app.post("/invoke")
async def invoke_model(request: Query, session_id="default_session"):
    """
    Endpoint to invoke the RAG model with a user query.

    Args:
        request (Query): The input query wrapped in a Pydantic model.

    Returns:
        dict: The generated response or error message.
    """
    REQUEST_COUNT.labels(endpoint="/invoke", method="POST").inc()
    with REQUEST_LATENCY.labels(endpoint="/invoke", method="POST").time():
        config = get_config_with_session(session_id)
        state = {
            "query": "",
            "context": "",
            "response": "",
            "messages": [],
            "summary": "",
            "user_name": ""
        }
        state["query"] = request.query

        logger.info("POST /invoke received with query: %s", state["query"])
        try:
            response = await get_response(state, config)
            logger.info("Response generated successfully for /invoke endpoint")
            return response
        except ValueError as e:
            ERROR_COUNT.labels(endpoint="/invoke", error_type="ValueError").inc()
            logger.error("ValueError during /invoke handling: %s", e, exc_info=True)
            return {"error": "Invalid input provided."}
        except RuntimeError as e:
            ERROR_COUNT.labels(endpoint="/invoke", error_type="RuntimeError").inc()
            logger.error("RuntimeError during /invoke handling: %s", e, exc_info=True)
            return {"error": "Internal processing error."}

@app.post("/webexhook")
async def webhook(request: Request):
    """
    Webhook endpoint for handling incoming Webex Teams messages.

    Args:
        request (Request): The incoming FastAPI request containing the webhook payload.

    Returns:
        dict: Status message or error message.
    """
    REQUEST_COUNT.labels(endpoint="/webexhook", method="POST").inc()
    with REQUEST_LATENCY.labels(endpoint="/webexhook", method="POST").time():
        try:
            payload = await request.json()
            logger.info("Webhook triggered with payload: %s", payload)
            data = payload.get("data", {})
            message_id = data.get("id")
            room_id = data.get("roomId")

            message = api.messages.get(message_id)
            user_email = message.personEmail
            config = get_config_with_session(room_id)
            state = {"query":message.text}
            logger.info("Webhook message from: %s, text: %s", user_email, state["query"])
            if user_email == "localhelper@webex.bot":
                logger.info("Ignoring bot's own message (loop prevention).")
                return {"message": "Ignoring bot's own message"}

            response = await get_response(state, config)
            logger.info("Responded to Webex message in room %s.", room_id)
            api.messages.create(roomId=room_id, text=response)
            return {"message": "Response sent"}

        except ValueError as e:
            ERROR_COUNT.labels(endpoint="/webexhook", error_type="ValueError").inc()
            logger.error("ValueError during Webex webhook processing: %s", e, exc_info=True)
            return {"error": "Invalid input provided."}
        except RuntimeError as e:
            ERROR_COUNT.labels(endpoint="/webexhook", error_type="RuntimeError").inc()
            logger.error("RuntimeError during Webex webhook processing: %s", e, exc_info=True)
            return {"error": "Internal processing error."}
        except (KeyError, AttributeError, TypeError) as e:
            ERROR_COUNT.labels(endpoint="/webexhook", error_type="KnownError").inc()
            logger.error("Known error during Webex webhook processing: %s", e, exc_info=True)
            return {"error": "Webhook processing failed due to malformed payload or missing data."}
