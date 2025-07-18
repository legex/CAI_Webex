"""
Module: vector_query.py

Provides vector similarity search and reranking for MongoDB in RAG pipelines,
optimized for technical support AI (e.g., Cisco/Webex queries).

Key Components:
    - VectorSearch: Handles $vectorSearch aggregation and cross-encoder reranking.

Usage Example:
    from datamanagement.db.vector_query import VectorSearch
    searcher = VectorSearch("mongodb://localhost:27017", "cisco_docs", "dataset")
    results = searcher.similarity_search("Webex issue?")
    ranked = searcher.rerank_results("Webex issue?", results)

Assumptions:
    - MongoDB with vector index on "response_embedding".
    - Documents include "thread_url", "query_chunk", "response_chunk".

Notes:
    - Integrates with embedding and scraping modules.
    - Thread-safe via model locks; monitor performance in production.
"""
from typing import List, Dict, Any
from datamanagement.core.querychunking import ChunkEmbedRank
from datamanagement.db.db_base import DBBase
from datamanagement.core.embedding_model import EmbeddingModel

class VectorSearch(DBBase):
    """
    Performs vector-based similarity search and reranking on a MongoDB collection.
    
    This class integrates with MongoDB's vector search capabilities to retrieve
    relevant documents based on query embeddings. It supports initial vector search
    followed by reranking using a cross-encoder model for improved relevance.
    Designed for use in RAG (Retrieval-Augmented Generation) pipelines, such as
    technical support conversational AI.
    
    Attributes:
        verbose (bool): If True, enables logging of key operations.
        top_k_vector (int): Number of top results to return from vector search.
        top_k_rerank (int): Number of top results after reranking.
        vector_index (str): Name of the MongoDB vector index to use.
        embedder (ChunkEmbedRank): Instance for generating query embeddings.
        model (CrossEncoder): Cross-encoder model for reranking.
        model_lock (threading.Lock): Lock for thread-safe model access.
        _embedding_cache (dict): Internal cache for query embeddings to avoid recomputation.
    
    Usage:
        searcher = VectorSearch(loginurl="mongodb://localhost:27017", 
        database="cisco_docs", collection="dataset")
        results = searcher.similarity_search("How to fix Webex connection?")
        ranked = searcher.rerank_results("How to fix Webex connection?", results)
    
    Note:
        - Assumes the MongoDB collection has a vector index (e.g., "vector_index") on the "response_embedding" field.
        - Requires fields like "thread_url", "query_chunk", "response_chunk" in documents.
        - Embeddings are cached per query for efficiency; clear the cache if needed.
    """
    def __init__(self, loginurl, database='', collection='',
                 top_k_vector: int = 50, top_k_rerank: int = 5,
                 verbose=True
                 ):
        """
        Initializes the VectorSearch instance with MongoDB connection and configuration.
        
        Args:
            loginurl (str): MongoDB connection URI (e.g., "mongodb://localhost:27017").
            database (str, optional): Database name. Defaults to ''.
            collection (str, optional): Collection name. Defaults to ''.
            top_k_vector (int, optional): Limit for vector search results. Defaults to 50.
            top_k_rerank (int, optional): Limit for reranked results. Defaults to 5.
            vector_index (str, optional): Name of the vector index in MongoDB. Defaults to "vector_index".
            verbose (bool, optional): Enable verbose logging. Defaults to True.
        """
        super().__init__(loginurl, database, collection)
        self.verbose = verbose
        self.top_k_vector = top_k_vector
        self.top_k_rerank = top_k_rerank
        self.embedder = ChunkEmbedRank()
        model_wrapper = EmbeddingModel.get_instance()
        self.model = model_wrapper.get_cross_encoder()
        self.model_lock = model_wrapper.lock
        self._embedding_cache = {}

    def get_embedded_query(self, query: str):
        """Generates or retrieves cached embedding for the query."""
        if query in self._embedding_cache:
            if self.verbose:
                print(f"Using cached embedding for query: {query[:50]}...")
            return self._embedding_cache[query]

        try:
            embedding = self.embedder.generate_embedding(query)
            self._embedding_cache[query] = embedding
            if self.verbose:
                print(f"Generated embedding for query: {query[:50]}...")
            return embedding
        except Exception as e:
            raise ValueError(f"Error generating query embedding: {str(e)}") from e

    def _pipeline(self, query: str):
        """
        returns pipeline for vector search
        """
        embedded_query = self.get_embedded_query(query)
        pipeline = [
            {
            "$vectorSearch": {
                    "queryVector": embedded_query,
                    "path": "response_embedding",
                    "numCandidates": 100,
                    "limit": self.top_k_vector,
                    "index": "vector_index"  # Replace with actual index name
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "thread_url": 1,
                    "query_chunk": 1,
                    "response_chunk": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        return pipeline

    def similarity_search(self, query: str):
        """
        Perform Similarity search to vector index
        """
        if self.verbose:
            print(f"Performing similarity search for query: {query[:50]}...")
        try:
            results = list(self.collection.aggregate(self._pipeline(query)))
            if not results and self.verbose:
                print("No results found from vector search.")
            return results
        except Exception as e:
            raise RuntimeError(f"Vector search failed: {str(e)}") from e

    def rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank search results using the cross-encoder.
        """
        if not results:
            if self.verbose:
                print("No results to rerank.")
            return []
        pairs = [(query, res["response_chunk"]) for res in results]

        if self.verbose:
            print(f"Reranking {len(results)} results...")
        try:
            with self.model_lock:
                scores = self.model.predict(pairs)

            for res, score in zip(results, scores):
                res['rerank_score'] = score

            ranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
            return ranked[:self.top_k_rerank]
        except Exception as e:
            raise RuntimeError(f"Reranking failed: {str(e)}") from e
