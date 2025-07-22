from threading import Lock
from sentence_transformers import SentenceTransformer, CrossEncoder
from datamanagement.core.utils import huggingface_login

class EmbeddingModel:
    _instance = None
    _lock = Lock()

    def __init__(self):
        huggingface_login()
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", cache_folder="./hf_cache")
        self.lock = Lock()
        self.cross_en = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get_model(self):
        return self.model
    
    def get_cross_encoder(self):
        return self.cross_en
