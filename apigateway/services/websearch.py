import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
#from langchain_core.messages import HumanMessage
from tavily import TavilyClient
from apigateway.prompt.prompt import TEMPLATE_CLEANDATA
from apigateway.services.modelbase import LLMModel
from datamanagement.cleanrawstring.cleanraw import clean_for_web_agent
from datamanagement.config.settings import INCLUDE_DOMAINS
from datamanagement.core.logger import setup_logger

load_dotenv(r'datamanagement\core\.env')
logger = setup_logger("websearch", 'datamanagement/log/websearchapi.log')

class WebSearch:
    def __init__(self):
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

    def tavilywrapper(self, query:str, top_k: int):
        try:
            logger.info("tavilywrapper called with query: %s, top_k: %s", query, top_k)
            result = self.tavily_client.search(query,
                                        max_results=top_k,
                                        include_raw_content=True,
                                        include_domains=INCLUDE_DOMAINS
                                        )
            logger.debug("tavilywrapper search results: %s", result)
            return result
        except Exception as e:
            logger.error("tavilywrapper failed for query '%s': %s", query, str(e))
            raise

    def search_web(self, query):
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
        try:
            logger.info("str_clean_wrapper called with %s items", len(raw_content))
            cleaned_items = []
            for i, item in enumerate(raw_content):
                try:
                    cleaned_item = clean_for_web_agent(item)
                    cleaned_items.append(cleaned_item)
                except ValueError as e:
                    logger.warning("Failed to clean item %s: %s", i, str(e))
                    continue
            
            logger.info("str_clean_wrapper cleaned %s out of %s items", len(cleaned_items), len(raw_content))
            return cleaned_items
        except Exception as e:
            logger.error("str_clean_wrapper failed: %s", str(e))
            raise

    def modelcall(self, query):
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
