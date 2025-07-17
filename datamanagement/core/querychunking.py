from datamanagement.core.generatebase import ChunkandGenerate
from datamanagement.core.embedding_model import EmbeddingModel
import numpy as np

class ChunkEmbedRank(ChunkandGenerate):
    def __init__(self, query):
        super().__init__()
        self.query = query
        model_wrapper = EmbeddingModel.get_instance()
        self.model = model_wrapper.get_model()
        self.cross_encoder = model_wrapper.get_cross_encoder()
        self.model_lock = model_wrapper.lock

    def generate_embedding(self):
        if len(self.query) >= 500:
            query_chunks = self.chunk_text(self.query)
        else:
            query_chunks = [self.query]
        
        with self.model_lock:
            query_embedding = [self.model.encode(chunk).tolist() for chunk in query_chunks]

        if len(query_embedding) == 1:
            return query_embedding[0]
        else:
            return np.mean(query_embedding, axis=0).tolist()
    