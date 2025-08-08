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

## Workflow:
- Webex Webhook event awaits for invokation
- Message received from webex to bot, and intent classifier will start its working to classify it as:
    - SmallTalk
    - RAG Query
- If SmallTalk:
    - LLM will be invoked with generic prompt Template

- If RAG Query:
    - Tool call will be initiated to perform vector and semantic search towards mongodb vector database
    - top_k = 5, docs will be retrieved and provided to the model as context
    - Model will generate response

- Store and summarize current conversation

## Future improvements:
- Implement web search with the bot
- Expand technical area of expertise
