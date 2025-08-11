import numpy as np
from datamanagement.core.generatebase import ChunkandGenerate
from datamanagement.core.embedding_model import EmbeddingModel
from datamanagement.core.logger import setup_logger

logger = setup_logger('chunk_embed_rank', 'datamanagement/log/chunk_embed_rank.log')

class ChunkEmbedRank(ChunkandGenerate):
    """
    Inherits text chunking and provides embedding generation methods,
    using a SentenceTransformer model and a cross-encoder for reranking.

    It handles:
    - Chunking the query if it exceeds a length threshold.
    - Generating embeddings for each chunk with thread-safe locking.
    - Returning a single embedding via averaging when multiple chunks exist.
    """
    def __init__(self):
        """
        Initialize ChunkEmbedRank by loading models from the singleton EmbeddingModel instance.
        Sets up model and cross-encoder references along with a threading lock.
        """
        super().__init__()
        logger.info("Initializing ChunkEmbedRank")
        model_wrapper = EmbeddingModel.get_instance()
        self.model = model_wrapper.get_model()
        self.cross_encoder = model_wrapper.get_cross_encoder()
        self.model_lock = model_wrapper.lock

    def generate_embedding(self, query: str = None, response_text = None):
        """
        Generate embedding(s) for the input query string.
        If query length exceeds 500 characters, it is chunked before embedding.

        Args:
            query (str, optional): The input text query to embed.

        Returns:
            list[float]: The embedding vector, either for single chunk or averaged for multiple chunks.
        """
        if not query:
            logger.warning("generate_embedding() called with empty or None query.")
            return None
        if len(query) >= 500:
            query_chunks = self.chunk_text(query)
            logger.info("Query length %d chars; split into %d chunks.",
                        len(query),
                        len(query_chunks)
                        )
        else:
            query_chunks = [query]
            logger.info("Query length %d chars; embedding directly without chunking.", len(query))

        with self.model_lock:
            query_embedding = [self.model.encode(chunk).tolist() for chunk in query_chunks]
            logger.debug("Generated embeddings for %d chunk(s).", len(query_embedding))

        if len(query_embedding) == 1:
            return query_embedding[0]
        else:
            logger.info("Averaged embedding across %d chunks.", len(query_embedding))
            return np.mean(query_embedding, axis=0).tolist()
