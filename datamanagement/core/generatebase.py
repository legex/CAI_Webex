from abc import ABC, abstractmethod
from langchain_text_splitters import RecursiveCharacterTextSplitter

class ChunkandGenerate(ABC):
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " "]
        )
                
    def chunk_text(self, text: str):
        return self.splitter.split_text(text)
    
    @abstractmethod
    def generate_embedding(self):
        pass
