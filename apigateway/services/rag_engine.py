"""
Module: rag_engine.py

Provides a Retrieval-Augmented Generation (RAG) engine that combines hybrid sparse and dense vector retrieval 
from a MongoDB backend with cross-encoder reranking and large language model (LLM) based answer generation.

Features:
- Integrates dense vector search ($vectorSearch) 
    and sparse full-text search ($text) on MongoDB collections.
- Implements a hybrid retrieval strategy that deduplicates and reranks results across sources.
- Prioritizes authoritative sources (e.g., help.webex.com) for context assembly.
- Applies heuristic filtering to exclude irrelevant or junk text chunks from context.
- Uses LangChain Ollama LLM bindings to generate precise,
    technical responses grounded on retrieved documents.
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
import re
import pymongo.errors
from dotenv import load_dotenv
from datamanagement.core.logger import setup_logger
from datamanagement.db.vector_query import VectorSearch


logger = setup_logger("rag_engine", 'datamanagement/log/rag_engine.log')

AUTHORITATIVE_DOMAINS = [
    "help.webex.com", "support.cisco.com", "www.webex.com"
]

class RagEngine:
    """
    Retrieval-Augmented Generation (RAG) engine for combining hybrid search over MongoDB with LLM-based response generation.
    """

    def __init__(self, database: str = 'cisco_docs',
                 collection: str = 'dataset',
                 ):
        """
        Initialize the RagEngine with MongoDB connection, vector search, and LLM model.

        Args:
            database (str): MongoDB database name.
            collection (str): MongoDB collection name.
        Raises:
            ValueError: If MONGO_URI is not set.
        """
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

        logger.info(
            "Initialized RagEngine for Vector Search"
        )

    def perform_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform dense vector similarity search.

        Args:
            query (str): The user query.

        Returns:
            List[Dict[str, Any]]: List of matching documents.
        """
        return self.vec_search.similarity_search(query)

    def perform_sparse_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform sparse (full-text) search.

        Args:
            query (str): The user query.

        Returns:
            List[Dict[str, Any]]: List of matching documents.
        """
        return self.vec_search.sparse_search(query)

    def perform_hybrid_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform hybrid search (dense + sparse).

        Args:
            query (str): The user query.

        Returns:
            List[Dict[str, Any]]: List of matching documents.
        """
        return self.vec_search.hybrid_search(query)

    def rerank_results(self, query: str,
                       search_results: List[Dict[str, Any]]
                       ) -> List[Dict[str, Any]]:
        """
        Rerank search results using cross-encoder or other reranking logic.

        Args:
            query (str): The user query.
            search_results (List[Dict[str, Any]]): Search results to rerank.

        Returns:
            List[Dict[str, Any]]: Reranked results.
        """
        return self.vec_search.rerank_results(query, search_results)

    def build_context(self, top_chunks: List[Dict[str, Any]]) -> str:
        """
        Build a context string from the top document chunks.

        Args:
            top_chunks (List[Dict[str, Any]]): Top-ranked document chunks.

        Returns:
            str: Concatenated context string.
        """
        technical_docs = [chunk['response_chunk'] for chunk in top_chunks]
        return "\n\n".join(technical_docs)

    def is_junk_chunk(self, text: str) -> bool:
        """
        Heuristically determine if a text chunk is junk/irrelevant.

        Args:
            text (str): The text chunk.

        Returns:
            bool: True if junk, False otherwise.
        """
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
        """
        Check if a URL is from an authoritative domain.

        Args:
            url (str): The URL to check.

        Returns:
            bool: True if authoritative, False otherwise.
        """
        return any(domain in url for domain in AUTHORITATIVE_DOMAINS)

    def get_full_threads(self,
                         reranked_chunks: List[Dict[str, Any]],
                         max_chunks_per_thread=2) -> Dict[str, str]:
        """
        Retrieve and assemble full threads for each unique thread URL,
        prioritizing authoritative sources.

        Args:
            reranked_chunks (List[Dict[str, Any]]): Reranked document chunks.
            max_chunks_per_thread (int): Max number of chunks per thread.

        Returns:
            Dict[str, str]: Mapping of thread URL to concatenated context.
        """
        thread_urls = {chunk['thread_url'] for chunk in reranked_chunks}
        logger.info("Collected thread URLs: %s", thread_urls)

        sorted_thread_urls = sorted(
            thread_urls,
            key=self.is_authoritative_url,
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
        except pymongo.errors.PyMongoError as e:
            logger.error("MongoDB fetch error in get_full_threads: %s", e)
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
        """
        Generate a response to the user query using hybrid retrieval and LLM.

        Args:
            query (str): The user query.

        Returns:
            str: The generated response or error message.
        """
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

            #response = self.chain.invoke({"technical_docs": context, "question": query})
            return context

        except pymongo.errors.ServerSelectionTimeoutError as e:
            logger.error("MongoDB server selection timeout: %s", e)
            return "Unable to connect to the database server. Please try again in a few moments."
        except pymongo.errors.ConnectionFailure as e:
            logger.error("MongoDB connection failure: %s", e)
            return "Database connection failed. Please check your network or try again later."
        except pymongo.errors.OperationFailure as e:
            logger.error("MongoDB operation failure: %s", e)
            return "An error occurred while querying the database. The query might be invalid or the index is missing."
        except (ValueError, KeyError, AttributeError) as e:
            logger.error("Unexpected error in RAG generation: %s", e)
            return f"Error generating response: {e}"
