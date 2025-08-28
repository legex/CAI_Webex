import os
import requests
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from tavily import TavilyClient
from coreservices.prompt.prompt import TEMPLATE_CLEANDATA
from coreservices.services.modelbase import LLMModel
from coreservices.services.settings import cleanraw, domains
from coreservices.logger.logger import setup_logger

load_dotenv(r'datamanagement\core\.env')
logger = setup_logger("websearch", 'log/websearchapi.log')

class WebSearch:
    def __init__(self):
        """
        Initialize WebSearch with Tavily client and LLM model.
        
        Raises:
            ValueError: If TAVILY_API_KEY is not found in environment variables.
            Exception: If initialization of Tavily client or LLM model fails.
        """
        try:
            logger.info("Initializing WebSearch")
            self.tavilyapikey = os.getenv("TAVILY_API_KEY")
            if not self.tavilyapikey:
                logger.error("TAVILY_API_KEY not found in environment variables")
                raise ValueError("TAVILY_API_KEY not found")

            self.tavily_client = TavilyClient(api_key=self.tavilyapikey)
            self.prompt = TEMPLATE_CLEANDATA
            _model_wrapper = LLMModel.get_instance()
            self.model = _model_wrapper.get_model()
            logger.info("WebSearch initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize WebSearch: %s", str(e))
            raise

    def tavilywrapper(self, query: str, top_k: int):
        """
        Wrapper for Tavily search API with error handling.
        
        Args:
            query (str): The search query to execute.
            top_k (int): Maximum number of search results to return.
            
        Returns:
            dict: Search results from Tavily API containing URLs and raw content.
            
        Raises:
            Exception: If the Tavily search API call fails.
        """
        try:
            logger.info("tavilywrapper called with query: %s, top_k: %s", query, top_k)
            url = domains
            response = requests.get(url, timeout=30)
            include_domains = response.json()
            result = self.tavily_client.search(query,
                                        max_results=top_k,
                                        include_raw_content=True,
                                        include_domains=include_domains["domains"]
                                        )
            logger.debug("tavilywrapper search results: %s", result)
            return result
        except Exception as e:
            logger.error("tavilywrapper failed for query '%s': %s", query, str(e))
            raise

    def search_web(self, query):
        """
        Search the web and extract URLs and raw content from results.
        
        Args:
            query (str): The search query to execute.
            
        Returns:
            dict: Dictionary containing 'urls' and 'raw_content' lists.
            
        Raises:
            Exception: If web search fails or result processing fails.
        """
        try:
            logger.info("search_web called with query: %s", query)
            search_results = self.tavilywrapper(query, 2)
            search_content = {'urls':[],
                        'raw_content': []
                        }
            for result in search_results['results']:
                search_content['urls'].append(result['url'])
                search_content['raw_content'].append(result['raw_content'])

            logger.info("search_web found %s results", len(search_content['urls']))
            logger.debug("search_web URLs: %s", search_content['urls'])
            return search_content
        except Exception as e:
            logger.error("search_web failed for query '%s': %s", query, str(e))
            raise

    def build_context_web(self, query):
        """
        Build context string from web search results by cleaning and joining content.
        
        Args:
            query (str): The search query to execute.
            
        Returns:
            str: Cleaned and joined web content as context string.
            
        Raises:
            Exception: If web search or content cleaning fails.
        """
        try:
            logger.info("build_context_web called with query: %s", query)
            web_content = self.search_web(query)
            cleaned_content = self.str_clean_wrapper(web_content['raw_content'])
            joined_string = "\n".join(cleaned_content)
            logger.info("build_context_web created context of length: %s", len(joined_string))
            return joined_string
        except Exception as e:
            logger.error("build_context_web failed for query '%s': %s", query, str(e))
            raise

    def str_clean_wrapper(self, raw_content: list):
        """
        Clean raw web content using the web agent cleaner.
        
        Args:
            raw_content (list): List of raw content strings to clean.
            
        Returns:
            list: List of cleaned content strings.
            
        Raises:
            Exception: If the cleaning process fails completely.
            
        Note:
            Individual item cleaning failures are logged as warnings but don't stop processing.
        """
        try:
            logger.info("str_clean_wrapper called with %s items", len(raw_content))
            cleaned_items = []
            for i, item in enumerate(raw_content):
                try:
                    #cleaned_item = clean_for_web_agent(item)
                    data = {"rawstrings":item}
                    url = cleanraw
                    response = requests.post(url, json=data, timeout=30)
                    cleaned_item = response.json()
                    cleaned_items.append(cleaned_item["cleaned_str"])
                except ValueError as e:
                    logger.warning("Failed to clean item %s: %s", i, str(e))
                    continue

            logger.info("str_clean_wrapper cleaned %s out of %s items",
                        len(cleaned_items),
                        len(raw_content))
            return cleaned_items
        except Exception as e:
            logger.error("str_clean_wrapper failed: %s", str(e))
            raise

    def modelcall(self, query):
        """
        Execute complete web search and LLM processing pipeline.
        
        This method:
        1. Builds web context from search results
        2. Formats the context using a prompt template
        3. Invokes the LLM model to process the data
        
        Args:
            query (str): The search query to execute.
            
        Returns:
            Any: Response from the LLM model after processing web content.
            
        Raises:
            Exception: If any step in the pipeline fails.
        """
        try:
            logger.info("modelcall called with query: %s", query)
            context = self.build_context_web(query)
            prompt = ChatPromptTemplate.from_template(self.prompt)
            formatted_prompt = prompt.format_prompt(raw_data=context
                                                    ).to_messages()
            logger.debug("modelcall formatted prompt: %s", formatted_prompt)
            model_response = self.model.invoke(formatted_prompt)
            logger.info("modelcall completed successfully")
            logger.debug("modelcall response: %s", model_response)
            return model_response
        except Exception as e:
            logger.error("modelcall failed for query '%s': %s", query, str(e))
            raise
