from typing import List
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import MessagesState
from apigateway.services.rag_engine import RagEngine
from apigateway.services.modelbase import LLMModel
from apigateway.prompt.prompt import TEMPLATE_Technical, TEMPLATE_General
from datamanagement.core.logger import setup_logger
import asyncio
import traceback

logger = setup_logger("langgraphtools", 'datamanagement/log/langgraphtools.log')

class Tools:
    """
    Defining tools for LLM calls
    Tools include:
    RAG Tool
    SmallTalk tool
    """
    def __init__(self):
        _model_wraper = LLMModel.get_instance()
        self.model = _model_wraper.get_model()
        self.prompt_technical = TEMPLATE_Technical
        self.prompt_general = TEMPLATE_General
        self.rg = RagEngine()
        logger.info("Tools class initialized with LLMModel and RagEngine.")

    def _returnprompt(self, template):
        try:
            logger.debug(f"Returning prompt from template: {template[:30]}...")
            return ChatPromptTemplate.from_template(template)
        except Exception as e:
            logger.error(f"Error constructing prompt from template: {e}")
            logger.debug(traceback.format_exc())
            raise

    def is_technical(self, query: str) -> bool:
        technical_keywords: List[str] = [
            "webex", "cucm", "cisco", "configure",
            "error", "deployment", "call manager",
            "troubleshoot", "installation"
        ]
        is_tech = any(word in query.lower() for word in technical_keywords)
        logger.info(f"is_technical('{query}') → {is_tech}")
        return is_tech

    def retrieval_tool(self, query: str) -> str:
        try:
            logger.info(f"[retrieval_tool] Running for query: {query!r}")
            result = self.rg.generate_response(query)
            logger.info(f"[retrieval_tool] Retrieved context length: {len(result) if result else 0}")
            return result
        except Exception as e:
            logger.error(f"[retrieval_tool] Exception during retrieval: {e}")
            logger.debug(traceback.format_exc())
            # Return empty string or fallback message to not break the flow
            return ""

    async def llm_with_context(self, query: str, context: str) -> str:
        try:
            logger.info(f"[llm_with_context] Query: {query!r} | Context length: {len(context) if context else 0}")
            prompt = self._returnprompt(TEMPLATE_Technical)
            formatted_prompt = (await prompt.aformat_prompt(
                technical_docs=context,
                question=query)).to_messages()
            logger.debug(f"[llm_with_context] Formatted prompt created, invoking model...")
            response = await self.model.ainvoke(formatted_prompt)
            logger.info(f"[llm_with_context] LLM response type: {type(response)}, response length: {len(response) if isinstance(response,str) else 'N/A'}")
            return response
        except Exception as e:
            logger.error(f"[llm_with_context] Exception during model invocation: {e}")
            logger.debug(traceback.format_exc())
            return "Sorry, I encountered an error processing your technical query."

    async def smalltalk_tool(self, query: str, summary=None) -> str:
        try:
            logger.info(f"[smalltalk_tool] Query: {query!r}, Summary provided: {'Yes' if summary else 'No'}")
            prompt = self._returnprompt(TEMPLATE_General)
            if summary:
                formatted_prompt = (await prompt.aformat_prompt(question=query, summary=summary)).to_messages()
            else:
                formatted_prompt = (await prompt.aformat_prompt(question=query, summary="no summary")).to_messages()
            logger.debug("[smalltalk_tool] Formatted prompt created, invoking model...")
            response = await self.model.ainvoke(formatted_prompt)
            logger.info(f"[smalltalk_tool] LLM response type: {type(response)}, response length: {len(response) if isinstance(response,str) else 'N/A'}")
            return response
        except Exception as e:
            logger.error(f"[smalltalk_tool] Exception during smalltalk model invocation: {e}")
            logger.debug(traceback.format_exc())
            return "Sorry, I'm having trouble chatting right now. Please try again later."

    def create_summary(self, summary_messages: list) -> str:
        try:
            logger.info(f"[create_summary] Called with {len(summary_messages)} messages.")
            result = self.model.invoke(summary_messages)
            logger.info(f"[create_summary] Summary generated of type {type(result)}, length: {len(result) if isinstance(result,str) else 'N/A'}")
            return result
        except Exception as e:
            logger.error(f"[create_summary] Exception during summary creation: {e}")
            logger.debug(traceback.format_exc())
            return "Error generating summary."

    async def routed_response(self, query: str) -> str:
        try:
            logger.info(f"[routed_response] Routing query: {query!r}")
            if self.is_technical(query):
                context = self.retrieval_tool(query)
                if not context.strip():
                    logger.warning(f"[routed_response] No context found for technical query: {query!r}")
                    return "Sorry, I could not find relevant documents for your question."
                response = await self.llm_with_context(query, context)
                logger.info(f"[routed_response] Technical response returned, length: {len(response) if isinstance(response,str) else 'N/A'}")
                return response
            else:
                response = await self.smalltalk_tool(query, summary=None)
                logger.info(f"[routed_response] Smalltalk response returned, length: {len(response) if isinstance(response,str) else 'N/A'}")
                return response
        except Exception as e:
            logger.error(f"[routed_response] Error during routing: {e}")
            logger.debug(traceback.format_exc())
            return "Sorry, I encountered an unexpected error while processing your query."
