from datamanagement.core.querychunking import ChunkEmbedRank
from datamanagement.db.db_base import DBBase
from datamanagement.core.embedding_model import EmbeddingModel

class VectorSearch(DBBase):
    def __init__(self, loginurl,
                 query, database='', collection='',
                 top_k_vector: int = 50, top_k_rerank: int = 5,
                 verbose=True
                 ):
        super().__init__(loginurl, database, collection)
        self.verbose = verbose
        self.query = query
        self.top_k_vector = top_k_vector
        self.top_k_rerank = top_k_rerank
        self.embedder = ChunkEmbedRank(self.query)
        self.embedded_query = self.embedder.generate_embedding()
        model_wrapper = EmbeddingModel.get_instance()
        self.model = model_wrapper.get_cross_encoder()
        self.model_lock = model_wrapper.lock

    @property
    def _embedded_query(self):
        return self.embedded_query
    
    def _pipeline(self):
        pipeline = [
            {
            "$vectorSearch": {
                    "queryVector": self._embedded_query,
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

    def similarity_search(self):
        return list(self.collection.aggregate(self._pipeline()))

    def rerank_results(self, results: list) -> list:
        """
        Rerank search results using the cross-encoder.
        """
        pairs = [(self.query, res["response_chunk"]) for res in results]

        with self.model_lock:
            scores = self.model.predict(pairs)

        for res, score in zip(results, scores):
            res['rerank_score'] = score

        ranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:self.top_k_rerank]
