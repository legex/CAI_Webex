from abc import ABC, abstractmethod
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datamanagement.core.logger import setup_logger

logger = setup_logger('chunk_and_generate', 'datamanagement/log/chunk_and_generate.log')

class ChunkandGenerate(ABC):
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " "]
        )
        logger.info(
            "Initialized the RecursiveCharacterTextSplitter with chunk_size=%d, chunk_overlap=%d",
            500,
            100)
          
    def chunk_text(self, text: str):
        logger.debug("Chunking text of length %d", len(text))
        return self.splitter.split_text(text)

    @abstractmethod
    def generate_embedding(self, query: str = None, response_text: str = None):
        logger.warning("Abstract method 'generate_embedding' not implemented")
        pass
