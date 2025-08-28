"""
Module: vector_query.py

Provides hybrid sparse and dense vector similarity search, with cross-encoder reranking, for use with MongoDB-based RAG pipelines. 
Optimized for technical support AI (e.g., Cisco/Webex queries).

Key Components:
    - VectorSearch: Unified interface for MongoDB $vectorSearch aggregation, $text full-text search, and post-retrieval reranking.

Usage Example:
    from db.vector_query import VectorSearch
    searcher = VectorSearch("mongodb://localhost:27017", "cisco_docs", "dataset")
    results = searcher.similarity_search("Webex issue?")
    ranked = searcher.rerank_results("Webex issue?", results)

Assumptions:
    - MongoDB is configured with a vector index on the "response_embedding" field for $vectorSearch queries.
    - Documents in the collection must include the fields: "thread_url", "query_chunk", "response_chunk".
    - A text index exists for $text-based sparse search (e.g., on "response_chunk").

Notes:
    - Integrates with embedding and scraping modules.
    - Thread-safe for model operations using locks.
    - Embedding caching prevents redundant computation per query.
    - Combine vector, sparse, and hybrid retrieval paths for optimal RAG recall and precision.
"""
from typing import List, Dict, Any
from core.querychunking import ChunkEmbedRank
from db.db_base import DBBase
from core.embedding_model import EmbeddingModel
from core.logger import setup_logger

logger = setup_logger('vector_search', 'datamanagement/log/vector_search.log')

class VectorSearch(DBBase):
    """
    Unified vector, sparse, and hybrid document retriever with reranking for MongoDB RAG pipelines.

    This search class supports dense retrieval with $vectorSearch and sparse query with $text. 
    Hybrid mode combines both, deduplicates by thread, then reranks using a cross-encoder.
    Suited for high-precision context retrieval in support and documentation bots.

    Attributes:
        verbose (bool): Enables logging of key operations to console.
        top_k_vector (int): Max results from vector search.
        top_k_rerank (int): Number of top results to retain after reranking.
        top_k_sparse (int): Max results from sparse full-text search.
        embedder (ChunkEmbedRank): Helper for query embedding.
        model (CrossEncoder): Cross-encoder for chunk reranking.
        model_lock (threading.Lock): Ensures thread safety for model predictions.
        _embedding_cache (dict): Cache for query embeddings.
    """
    def __init__(self, loginurl,
                 database='', collection='',
                 top_k_vector: int = 50, top_k_rerank: int = 5,
                 top_k_sparse: int = 20, verbose=True
                 ):
        """
        Initialize the search instance with MongoDB connection and RAG retrieval configuration.

        Args:
            loginurl (str): MongoDB connection URI, e.g. "mongodb://localhost:27017".
            database (str, optional): Database name. Defaults to ''.
            collection (str, optional): Collection name. Defaults to ''.
            top_k_vector (int, optional): Max results for vector search. Defaults to 50.
            top_k_rerank (int, optional): Max results after reranking. Defaults to 5.
            top_k_sparse (int, optional): Max results for $text sparse search. Defaults to 20.
            verbose (bool, optional): Enable verbose console logging. Defaults to True.
        """
        super().__init__(loginurl, database, collection)
        self.verbose = verbose
        self.top_k_vector = top_k_vector
        self.top_k_rerank = top_k_rerank
        self.top_k_sparse = top_k_sparse
        self.embedder = ChunkEmbedRank()
        model_wrapper = EmbeddingModel.get_instance()
        self.model = model_wrapper.get_cross_encoder()
        self.model_lock = model_wrapper.lock
        self._embedding_cache = {}

    def get_embedded_query(self, query: str):
        """
        Returns the embedding vector for the input query, using cache if available.

        Args:
            query (str): Input search query.

        Returns:
            list[float]: Query embedding vector.
        """
        if query in self._embedding_cache:
            logger.debug("Using cached embedding for query: %s...", query[:50])
            if self.verbose:
                print(f"Using cached embedding for query: {query[:50]}...")
            return self._embedding_cache[query]
        try:
            embedding = self.embedder.generate_embedding(query)
            self._embedding_cache[query] = embedding
            logger.info("Generated embedding for query: %s (embedding shape: %d)", query[:50], len(embedding))
            if self.verbose:
                print(f"Generated embedding for query: {query[:50]}...")
            return embedding
        except Exception as e:
            logger.error("Error generating query embedding: %s", str(e))
            raise ValueError(f"Error generating query embedding: {str(e)}") from e

    def _pipeline(self, query: str):
        """
        Constructs the vector search aggregation pipeline for the input query.

        Args:
            query (str): Query string.

        Returns:
            list[dict]: MongoDB aggregation pipeline for vector search.
        """
        embedded_query = self.get_embedded_query(query)
        pipeline = [
            {
                "$vectorSearch": {
                    "queryVector": embedded_query,
                    "path": "response_embedding",
                    "numCandidates": 500,
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

    def similarity_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve documents using MongoDB vector similarity search.

        Args:
            query (str): Search query.

        Returns:
            list of dict: Top-K candidate chunks from vector search.
        """
        logger.info("Vector search started for query: '%s'", query[:60])
        if self.verbose:
            print(f"Performing similarity search for query: {query[:50]}...")
        try:
            results = list(self.collection.aggregate(self._pipeline(query)))
            logger.info("Vector search found %d results for query: '%s'", len(results), query[:60])
            if not results and self.verbose:
                logger.warning("No results found from vector search.")
                print("No results found from vector search.")
            return results
        except Exception as e:
            logger.error("Vector search failed: %s", str(e))
            raise RuntimeError(f"Vector search failed: {str(e)}") from e

    def sparse_search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Perform full-text (sparse) search using MongoDB's $text operator.

        Args:
            query (str): Search query.
            top_k (int, optional): Max results to return. If None, use self.top_k_sparse.

        Returns:
            list of dict: Top-K candidate chunks from full-text search.
        """
        logger.info("Sparse text search started for query: '%s' (top_k=%d)", query[:60], top_k or self.top_k_sparse)
        if top_k is None:
            top_k = self.top_k_sparse
        if self.verbose:
            print(f"Performing sparse (text) search for query: {query[:50]}...")
        try:
            cursor = self.collection.find(
                {"$text": {"$search": query}},
                {
                    "_id": 0,
                    "thread_url": 1,
                    "query_chunk": 1,
                    "response_chunk": 1,
                    "score": {"$meta": "textScore"}
                }
            ).sort([("score", {"$meta": "textScore"})]).limit(top_k)
            results = list(cursor)
            logger.info("Sparse search returned %d results for query: '%s'", len(results), query[:60])
            if not results and self.verbose:
                logger.warning("No results found from sparse search.")
                print("No results found from sparse search.")
            return results
        except Exception as e:
            logger.error("Sparse search failed: %s", str(e))
            raise RuntimeError(f"Sparse (full-text) search failed: {str(e)}") from e

    def rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank results using a cross-encoder model.

        Args:
            query (str): Input search query.
            results (List[Dict[str, Any]]): List of candidate chunks.

        Returns:
            List[Dict[str, Any]]: Top reranked candidate chunks.
        """
        if not results:
            logger.warning("No results to rerank.")
            if self.verbose:
                print("No results to rerank.")
            return []
        logger.info("Reranking %d results for query: '%s'", len(results), query[:60])
        pairs = [(query, res["response_chunk"]) for res in results]
        if self.verbose:
            print(f"Reranking {len(results)} results...")
        try:
            with self.model_lock:
                scores = self.model.predict(pairs)
            for res, score in zip(results, scores):
                res['rerank_score'] = score
            ranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
            logger.info("Reranking complete, returning top %d results.", self.top_k_rerank)
            return ranked[:self.top_k_rerank]
        except Exception as e:
            logger.error("Reranking failed: %s", str(e))
            raise RuntimeError(f"Reranking failed: {str(e)}") from e

    def hybrid_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Hybrid search: combine vector and sparse search results, deduplicate by 'thread_url',
        keep only the highest-scoring chunk per thread, then rerank the unique chunks.

        Args:
            query (str): Search query.

        Returns:
            List[Dict[str, Any]]: Top reranked candidate chunks per unique thread.
        """
        logger.info("Hybrid search started for query: '%s'", query[:60])
        if self.verbose:
            print(f"Running hybrid search for query: {query[:50]}...")
        vector_results = self.similarity_search(query)
        sparse_results = self.sparse_search(query)
        logger.debug("Vector results: %d, Sparse results: %d", len(vector_results), len(sparse_results))

        # Collect highest-scored chunk per thread_url, considering both sources
        best_per_thread = {}

        def get_score(res):
            return res.get("rerank_score", res.get("score", 0))

        for res in vector_results + sparse_results:
            thread_url = res.get("thread_url")
            score = get_score(res)
            if thread_url:
                if (
                    thread_url not in best_per_thread
                    or get_score(best_per_thread[thread_url]) < score
                ):
                    best_per_thread[thread_url] = res

        combined_results = list(best_per_thread.values())
        logger.info("Hybrid search deduplicated to %d unique threads.", len(combined_results))
        reranked = self.rerank_results(query, combined_results)
        logger.info("Hybrid search reranking complete for query: '%s'", query[:60])
        logger.debug("Hybrid search threads considered: %s", [res.get("thread_url") for res in combined_results])
        if self.verbose:
            print(
                "Hybrid debug—Threads considered:",
                [res.get("thread_url") for res in combined_results],
            )
        return reranked
