# WRAITH

## Overview:
An AI Agent capable to solving complex technical queries which revolves around Telco vendor ( Cisco )

## Architecture:
- UI : Webex Teams App
- API Router: FastAPI
- LLM Orchestrator: Langgraph
- LLM Model: mistral
- Embedding Model: allmpnet
- Vector Database: MongoDB

## Microservice Architecture:
Containatainarization is divided as:
- Datamanagement handles:
    - Data ingestion
        - Chunking
        - Embedding
        - Save to db
    - RAG Engine:
        - Query embedding
        - Vector search
        - semantic search
- Coreservices handles:
    - LLM response generation
    - Workflow orchestration
    - Gateway for UI
- LLM-serve:
    - serves LLM model for coreservices

## Workflow:
- Webex Webhook event awaits for invokation
- Message received from webex to bot, and intent classifier will start its working to classify it as:
    - SmallTalk
    - RAG Query
- If SmallTalk:
    - LLM will be invoked with generic prompt Template

- If RAG Query:
    - Tool call will be initiated to perform vector and semantic search towards mongodb vector database
    - Parallel websearch with user query will be made, and context will be generated based on vector search and web search results
    - top_k = 5, docs will be retrieved and provided to the model as context, top_k=2 for websearch
    - Model will generate response

- Store and summarize current conversation

## Future improvements:
- Small features to implement:
    - SLM or distilled for intent recognition and NER
- Implement meeting scheduler integrated (use existing meeting proj code)
- Expand technical area of expertise (limited due MongogDB size limit)
