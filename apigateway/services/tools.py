from typing import List
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import MessagesState
from apigateway.services.rag_engine import RagEngine
from apigateway.services.modelbase import LLMModel
from apigateway.prompt.prompt import TEMPLATE_Technical, TEMPLATE_General
from datamanagement.core.logger import setup_logger
import traceback

logger = setup_logger("langgraphtools", 'datamanagement/log/langgraphtools.log')

class State(MessagesState):
    summary: str

class Tools:
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
            logger.error(f"Error constructing prompt: {e}")
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
            logger.error(f"[retrieval_tool] Exception: {e}")
            logger.debug(traceback.format_exc())
            return ""

    async def llm_with_context(self, messages: List[BaseMessage], context: str, summary: str = "", username: str = "") -> str:
        try:
            logger.info(f"[llm_with_context] Messages length: {len(messages)}, context length: {len(context) if context else 0}")
            prompt = self._returnprompt(self.prompt_technical)
            formatted_prompt = (await prompt.aformat_prompt(
                technical_docs=context,
                conversation_summary=summary,
                messages=messages,
                username=username
            )).to_messages()
            logger.debug(f"[llm_with_context] Prompt formatted; invoking model.")
            response = await self.model.ainvoke(formatted_prompt)
            logger.info(f"[llm_with_context] Received response of length {len(response) if isinstance(response, str) else 'N/A'}")
            return response
        except Exception as e:
            logger.error(f"[llm_with_context] Exception: {e}")
            logger.debug(traceback.format_exc())
            return "Sorry, I encountered an error processing your technical query."

    async def smalltalk_tool(self, messages: List[BaseMessage], summary=None, username:str ="") -> str:
        try:
            logger.info(f"[smalltalk_tool] Messages length: {len(messages)}, Summary present: {'Yes' if summary else 'No'}")
            prompt = self._returnprompt(self.prompt_general)
            if summary:
                # Assume summary variable mapped correctly in your TEMPLATE_General prompt
                formatted_prompt = (await prompt.aformat_prompt(
                    message=messages[-1].content,
                    summary=summary,
                    username=username
                )).to_messages()
            else:
                formatted_prompt = (await prompt.aformat_prompt(
                    message=messages[-1].content,
                    summary="no summary",
                    username=username
                )).to_messages()
            logger.debug(f"[smalltalk_tool] Prompt formatted; invoking model.")
            response = await self.model.ainvoke(formatted_prompt)
            logger.info(f"[smalltalk_tool] Received response of length {len(response) if isinstance(response, str) else 'N/A'}")
            return response
        except Exception as e:
            logger.error(f"[smalltalk_tool] Exception: {e}")
            logger.debug(traceback.format_exc())
            return "Sorry, I'm having trouble chatting right now."

    def create_summary(self, summary_messages: List[BaseMessage]) -> str:
        try:
            logger.info(f"[create_summary] Called with {len(summary_messages)} messages")
            result = self.model.invoke(summary_messages)
            logger.info(f"[create_summary] Summary generated with length {len(result) if isinstance(result, str) else 'N/A'}")
            return result
        except Exception as e:
            logger.error(f"[create_summary] Exception: {e}")
            logger.debug(traceback.format_exc())
            return "Error generating summary."

    async def routed_response(
        self,
        query: str,
        messages: List[BaseMessage] = None,
        summary: str = "",
        username: str = ""
    ) -> str:
        try:
            logger.info(f"[routed_response] Routing query: {query!r}")
            messages = messages or [HumanMessage(content=query)]

            if self.is_technical(query):
                context = self.retrieval_tool(query)
                if not context.strip():
                    logger.warning(f"[routed_response] No context found for technical query: {query!r}")
                    return "Sorry, I could not find relevant documents for your question."
                # Pass full messages, context, summary, user_name
                response = await self.llm_with_context(messages, context=context, summary=summary, username=username)
                logger.info(f"[routed_response] Technical response returned")
                return response
            else:
                response = await self.smalltalk_tool(messages, summary=summary, username=username)
                logger.info(f"[routed_response] Smalltalk response returned")
                return response
        except Exception as e:
            logger.error(f"[routed_response] Error: {e}")
            logger.debug(traceback.format_exc())
            return "Sorry, I encountered an unexpected error while processing your query."

