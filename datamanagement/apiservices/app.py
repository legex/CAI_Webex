from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST
    )
from fastapi import FastAPI
from fastapi.responses import Response
from core.logger import setup_logger
from cleanrawstring.cleanraw import clean_for_web_agent
from apiservices.parastruct import CleanRaw, Query
from core.rag_engine import RagEngine
from config.settings import INCLUDE_DOMAINS

logger = setup_logger("datasetvice", 'log/datasetvice.log')

dataservice = FastAPI()
rg = RagEngine()

REQUEST_COUNT = Counter(
    "api_requests_total", "Total API Requests", ["endpoint", "method"]
)
ERROR_COUNT = Counter(
    "api_errors_total", "Total API Errors", ["endpoint", "error_type"]
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds", "API Request latency in seconds", ["endpoint", "method"]
)

@dataservice.get("/metrics")
def metrics():
    """
    Expose Prometheus metrics.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@dataservice.get("/")
def root():
    """
    Root endpoint for health check or basic info.
    """
    REQUEST_COUNT.labels(endpoint="/", method="GET").inc()
    with REQUEST_LATENCY.labels(endpoint="/", method="GET").time():
        logger.info("Root endpoint called.")
        return {"message": "This is API router for Chatbot"}

@dataservice.post("/cleanraw")
def cleanraw(request: CleanRaw):
    REQUEST_COUNT.labels(endpoint="/cleanraw", method="POST").inc()
    with REQUEST_LATENCY.labels(endpoint="/cleanraw", method="POST").time():
        str_retrieved = request.rawstrings
        try:
            clean_str = clean_for_web_agent(str_retrieved)
            return {"cleaned_str":clean_str}
        except ValueError as e:
            ERROR_COUNT.labels(endpoint="/cleanraw", error_type="ValueError").inc()
            logger.error("ValueError during /cleanraw handling: %s", e, exc_info=True)
            return {"error": "Invalid input provided."}
        except RuntimeError as e:
            ERROR_COUNT.labels(endpoint="/cleanraw", error_type="RuntimeError").inc()
            logger.error("RuntimeError during /cleanraw handling: %s", e, exc_info=True)
            return {"error": "Internal processing error."}


@dataservice.get("/domains")
def getdomains():
    REQUEST_COUNT.labels(endpoint="/domains", method="GET").inc()
    with REQUEST_LATENCY.labels(endpoint="/domains", method="GET").time():
        try:
            return {"domains": INCLUDE_DOMAINS}
        except ValueError as e:
            ERROR_COUNT.labels(endpoint="/domains", error_type="ValueError").inc()
            logger.error("ValueError during /domains handling: %s", e, exc_info=True)
            return {"error": "Invalid input provided."}
        except RuntimeError as e:
            ERROR_COUNT.labels(endpoint="/domains", error_type="RuntimeError").inc()
            logger.error("RuntimeError during /domains handling: %s", e, exc_info=True)
            return {"error": "Internal processing error."}

@dataservice.post("/ragengine")
def Rag_Engine(request: Query):
    REQUEST_COUNT.labels(endpoint="/ragengine", method="POST").inc()
    with REQUEST_LATENCY.labels(endpoint="/ragengine", method="POST").time():
        query_retrieved = request.query
        try:
            context = rg.generate_response(query_retrieved)
            return {"context": context}
        except ValueError as e:
            ERROR_COUNT.labels(endpoint="/ragengine", error_type="ValueError").inc()
            logger.error("ValueError during /ragengine handling: %s", e, exc_info=True)
            return {"error": "Invalid input provided."}
        except RuntimeError as e:
            ERROR_COUNT.labels(endpoint="/ragengine", error_type="RuntimeError").inc()
            logger.error("RuntimeError during /ragengine handling: %s", e, exc_info=True)
            return {"error": "Internal processing error."}
