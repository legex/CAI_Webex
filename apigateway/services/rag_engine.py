"""
Module: rag_engine.py

Provides a Retrieval-Augmented Generation (RAG) engine that combines hybrid sparse and dense vector retrieval 
from a MongoDB backend with cross-encoder reranking and large language model (LLM) based answer generation.

Features:
- Integrates dense vector search ($vectorSearch) and sparse full-text search ($text) on MongoDB collections.
- Implements a hybrid retrieval strategy that deduplicates and reranks results across sources.
- Prioritizes authoritative sources (e.g., help.webex.com) for context assembly.
- Applies heuristic filtering to exclude irrelevant or junk text chunks from context.
- Uses LangChain Ollama LLM bindings to generate precise, technical responses grounded on retrieved documents.
- Supports detailed error handling for MongoDB connection and query failures.
- Logs detailed operational info for tracing and debugging.

Typical usage:
rag = RagEngine()
response = rag.generate_response("How to manually add a user in Webex Control Hub?")
print(response)

Assumptions:
- MongoDB collection contains documents with embedded vectors and full-text indexes.
- Documents include fields such as 'thread_url', 'query_chunk', 'response_chunk'.
- An LLM model (e.g., Mistral) is available via Ollama bindings.
- Environment variable `MONGO_URI` is set for MongoDB connection.

This module is designed for robust, scalable technical support conversational AI use cases,
where combining diverse retrieval modes improves answer completeness and accuracy.
"""
from typing import List, Dict, Any
import os
import pymongo.errors
from dotenv import load_dotenv
import json
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from datamanagement.db.vector_query import VectorSearch
from apigateway.prompt.prompt import TEMPLATE
import re
from datamanagement.core.logger import setup_logger


logger = setup_logger("rag_engine", 'datamanagement/log/rag_engine.log')

AUTHORITATIVE_DOMAINS = [
    "help.webex.com", "support.cisco.com", "www.webex.com"
]

class RagEngine:
    """
    Retrieval-Augmented Generation (RAG) engine for combining hybrid search over MongoDB with LLM-based response generation.
    """

    def __init__(self, database: str = 'cisco_docs', collection: str = 'dataset',
                 model_name: str = 'mistral', temperature: float = 0.0):
        load_dotenv(dotenv_path=r'datamanagement/core/.env')
        self.mongo_uri = os.getenv("MONGO_URI")
        if not self.mongo_uri:
            logger.error("MONGO_URI must be set in .env file.")
            raise ValueError("MONGO_URI must be set in .env file.")

        self.database = database
        self.collection = collection
        self.vec_search = VectorSearch(
            loginurl=self.mongo_uri,
            database=self.database,
            collection=self.collection,
            verbose=False
        )
        self.model = OllamaLLM(model=model_name, temperature=temperature)
        prompt = ChatPromptTemplate.from_template(TEMPLATE)
        self.chain = prompt | self.model

        logger.info(f"Initialized RagEngine with model '{model_name}', database '{database}', collection '{collection}'.")

    def perform_search(self, query: str) -> List[Dict[str, Any]]:
        return self.vec_search.similarity_search(query)

    def perform_sparse_search(self, query: str) -> List[Dict[str, Any]]:
        return self.vec_search.sparse_search(query)

    def perform_hybrid_search(self, query: str) -> List[Dict[str, Any]]:
        return self.vec_search.hybrid_search(query)

    def rerank_results(self, query: str, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self.vec_search.rerank_results(query, search_results)

    def build_context(self, top_chunks: List[Dict[str, Any]]) -> str:
        technical_docs = [chunk['response_chunk'] for chunk in top_chunks]
        return "\n\n".join(technical_docs)

    def is_junk_chunk(self, text: str) -> bool:
        if not text or len(text.strip()) < 30:
            return True
        if len(re.sub(r"[a-zA-Z0-9]", "", text)) / max(1, len(text)) > 0.5:
            return True
        junk_patterns = [
            r"\b(magyar|polski|čeština|norsk|hrvatski|srpski|עברית|suomi|slovenščina|slovenský|欢迎|sign[ -]?in|home[ /])\b",
            r"^(?:[^\w\s]{10,})$",
            r"\b(site map|language:|privacy|copyright|article add users|manage external users)\b"
        ]
        for pat in junk_patterns:
            if re.search(pat, text, re.IGNORECASE):
                return True
        return False

    def is_authoritative_url(self, url: str) -> bool:
        return any(domain in url for domain in AUTHORITATIVE_DOMAINS)

    def get_full_threads(self, reranked_chunks: List[Dict[str, Any]], max_chunks_per_thread=2) -> Dict[str, str]:
        thread_urls = {chunk['thread_url'] for chunk in reranked_chunks}
        logger.info(f"Collected thread URLs: {thread_urls}")

        # Prioritize authoritative URLs first
        sorted_thread_urls = sorted(
            thread_urls,
            key=lambda url: self.is_authoritative_url(url),
            reverse=True
        )
        reranked_map = {url: [] for url in thread_urls}
        for chunk in reranked_chunks:
            url = chunk['thread_url']
            if url in reranked_map:
                reranked_map[url].append(chunk['response_chunk'])

        try:
            all_thread_chunks = {
                url: [doc['response_chunk'] for doc in self.vec_search.collection.find(
                    {"thread_url": url},
                    {"_id": 0, "response_chunk": 1}
                ).limit(5)]
                for url in thread_urls
            }
        except Exception as e:
            logger.error(f"MongoDB fetch error in get_full_threads: {e}")
            all_thread_chunks = {}

        thread_contexts = {}
        for url in sorted_thread_urls:
            candidate_chunks = reranked_map.get(url, [])
            filtered = [c for c in candidate_chunks if not self.is_junk_chunk(c)]
            if len(filtered) < max_chunks_per_thread:
                additional = [
                    c for c in all_thread_chunks.get(url, [])
                    if not self.is_junk_chunk(c) and c not in filtered
                ]
                filtered += additional
            thread_contexts[url] = "\n\n".join(filtered[:max_chunks_per_thread])

        return thread_contexts

    def generate_response(self, query: str) -> str:
        if not query:
            return "Query cannot be empty."
        try:
            search_results = self.perform_hybrid_search(query)
            if not search_results:
                logger.info("No relevant documents found.")
                return "No relevant documents found."

            top_chunks = self.rerank_results(query, search_results)
            thread_contexts = self.get_full_threads(top_chunks, max_chunks_per_thread=2)

            if thread_contexts:
                context = "\n\n".join([ctx for ctx in thread_contexts.values() if ctx.strip()])
            else:
                context = self.build_context(top_chunks)

            response = self.chain.invoke({"technical_docs": context, "question": query})
            return response

        except pymongo.errors.ServerSelectionTimeoutError as e:
            logger.error(f"MongoDB server selection timeout: {e}")
            return "Unable to connect to the database server. Please try again in a few moments."
        except pymongo.errors.ConnectionFailure as e:
            logger.error(f"MongoDB connection failure: {e}")
            return "Database connection failed. Please check your network or try again later."
        except pymongo.errors.OperationFailure as e:
            logger.error(f"MongoDB operation failure: {e}")
            return "An error occurred while querying the database. The query might be invalid or the index is missing."
        except Exception as e:
            logger.error(f"Unexpected error in RAG generation: {e}")
            return f"Error generating response: {e}"
