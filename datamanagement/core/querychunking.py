from datamanagement.core.generatebase import ChunkandGenerate
from datamanagement.core.embedding_model import EmbeddingModel
import numpy as np
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

    def generate_embedding(self, query: str = None):
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
            logger.info(f"Query length {len(query)} chars; split into {len(query_chunks)} chunks.")
        else:
            query_chunks = [query]
            logger.info(f"Query length {len(query)} chars; embedding directly without chunking.")

        with self.model_lock:
            query_embedding = [self.model.encode(chunk).tolist() for chunk in query_chunks]
            logger.debug(f"Generated embeddings for {len(query_embedding)} chunk(s).")

        if len(query_embedding) == 1:
            return query_embedding[0]
        else:
            logger.info(f"Averaged embedding across {len(query_embedding)} chunks.")
            return np.mean(query_embedding, axis=0).tolist()
