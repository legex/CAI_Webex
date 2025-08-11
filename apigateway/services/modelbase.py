from langchain_ollama import OllamaLLM

class LLMModel:
    _instance = None
    
    def __init__(self, model_name: str = 'mistral', temperature: float = 0.0):
        self.llmmodel = OllamaLLM(model=model_name, temperature=temperature, num_ctx= 8192)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_model(self):
        return self.llmmodel
