import os
import json
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from tavily import TavilyClient
from apigateway.prompt.prompt import TEMPLATE_CLEANDATA
from apigateway.services.modelbase import LLMModel
from datamanagement.cleanrawstring.cleanraw import clean_for_web_agent

load_dotenv(r'datamanagement\core\.env')
include_domains= ["cisco.com", "webex.com"]
apikey = os.getenv("TAVILY_API_KEY")
prompt = ChatPromptTemplate.from_template(TEMPLATE_CLEANDATA)

_model_wraper = LLMModel.get_instance()
model = _model_wraper.get_model()

collected_urls = []

tavily_client = TavilyClient(api_key=apikey)
urls = tavily_client.search("cisco sip error 403", max_results=2, include_domains=include_domains)
for result in urls['results']:
    collected_urls.append(result['url'])

response = tavily_client.extract(collected_urls)
end_response=""""""

for content in response['results']:
    with open("test.json","a") as f:
        json.dump(content, f)
    formatted_prompt = prompt.format_prompt(
        raw_data=content['raw_content']
        ).to_messages()
    model_response = model.invoke(formatted_prompt)
    end_response += model_response

print(end_response)
