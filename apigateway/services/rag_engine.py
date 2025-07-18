
from typing import List, Dict, Any
import os
import logging
import pymongo.errors
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from datamanagement.db.vector_query import VectorSearch
from apigateway.prompt.prompt import TEMPLATE

class RagEngine:
    def __init__(self, database: str = 'cisco_docs', collection: str = 'dataset',
                  model_name: str = 'llama3.2:1b', temperature:float = 0.0):

        load_dotenv(dotenv_path=r'datamanagement\core\.env')
        self.mongo_uri = os.getenv("MONGO_URI")
        if not self.mongo_uri:
            raise ValueError("MONGO_URI must be set in .env file.")
        self.database = database
        self.collection = collection
        self.vec_search = VectorSearch(
            loginurl=self.mongo_uri,
            database=self.database,
            collection=self.collection
        )
        self.model = OllamaLLM(model=model_name, temperature=temperature)
        prompt = ChatPromptTemplate.from_template(TEMPLATE)
        self.chain = prompt | self.model

    def perform_search(self, query: str) -> List[Dict[str, Any]]:
        """Performs vector similarity search using the reusable VectorSearch instance."""
        return self.vec_search.similarity_search(query)

    def rerank_results(self, query: str, 
                       search_results: List[Dict[str, Any]]
                       ) -> List[Dict[str, Any]]:
        """Reranks the search results using the reusable VectorSearch instance."""
        return self.vec_search.rerank_results(query, search_results)

    def build_context(self, top_chunks: List[Dict[str, Any]]) -> str:
        """Builds a context string from the top reranked chunks."""
        technical_docs = [chunk['response_chunk'] for chunk in top_chunks]
        return "\n\n".join(technical_docs)

    def generate_response(self, query: str) -> str:
        """
        Generates a response for the given query using the full RAG pipeline.
        
        Args:
            query (str): The user's query string.
        
        Returns:
            str: The generated response or an error message.
        """
        if not query:
            return "Query cannot be empty."

        try:
            search_results = self.perform_search(query)
            if not search_results:
                return "No relevant documents found."

            top_chunks = self.rerank_results(query, search_results)
            context = self.build_context(top_chunks)

            response = self.chain.invoke({"technical_docs": context, "question": query})
            return response
        except pymongo.errors.ServerSelectionTimeoutError as e:
            logging.error(f"MongoDB server selection timeout: {str(e)}")
            return "Unable to connect to the database server. Please try again in a few moments."

        except pymongo.errors.ConnectionFailure as e:
            logging.error(f"MongoDB connection failure: {str(e)}") 
            return "Database connection failed. Please check your network or try again later."

        except pymongo.errors.OperationFailure as e:
            logging.error(f"MongoDB operation failed: {str(e)}")
            return "An error occurred while querying the database. The query might be invalid or the index is missing."

        except Exception as e:
            logging.error(f"Unexpected error in RAG generation: {str(e)}")
            return f"Error generating response: {str(e)}"