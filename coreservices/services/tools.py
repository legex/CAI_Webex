"""
Tools module for handling prompt construction, technical/smalltalk routing, retrieval, and summary generation
using LLMs and RAG engine for the CAI_Webex API Gateway.
"""
import requests
from typing import List
import traceback
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage, HumanMessage
from coreservices.services.modelbase import LLMModel
from coreservices.prompt.prompt import TEMPLATE_TECHNICAL, TEMPLATE_GENERAL, TEMPLATE_SUMMARY
from coreservices.services.settings import rag_apiendpoint
from datamanagement.core.logger import setup_logger


logger = setup_logger("langgraphtools", 'datamanagement/log/langgraphtools.log')

class Tools:
    """Provides tools for prompt handling, technical/smalltalk routing, and summary creation."""
    def __init__(self):
        """
        Initialize the Tools class with model, prompts, and retrieval engine.
        """
        _model_wraper = LLMModel.get_instance()
        self.model = _model_wraper.get_model()
        self.prompt_technical = TEMPLATE_TECHNICAL
        self.prompt_general = TEMPLATE_GENERAL
        self.promt_summary = TEMPLATE_SUMMARY
        self.rgurl = rag_apiendpoint
        logger.info("Tools class initialized with LLMModel and RagEngine.")

    def _returnprompt(self, template):
        """
        Returns a ChatPromptTemplate from the given template string.

        Args:
            template (str): The prompt template string.

        Returns:
            ChatPromptTemplate: The constructed prompt template.
        """
        try:
            logger.debug("Returning prompt from template: %.30s...", template)
            return ChatPromptTemplate.from_template(template)
        except (ValueError, TypeError) as e:
            logger.error("Error constructing prompt: %s", e)
            logger.debug("%s", traceback.format_exc())
            raise

    def is_technical(self, query: str) -> bool:
        """
        Determines if the query is technical based on keywords.

        Args:
            query (str): The user's query.

        Returns:
            bool: True if technical, False otherwise.
        """
        technical_keywords: List[str] = [
            "webex", "cucm", "cisco", "configure",
            "error", "deployment", "call manager",
            "troubleshoot", "installation"
        ]
        is_tech = any(word in query.lower() for word in technical_keywords)
        logger.info("is_technical('%s') → %s", query, is_tech)
        return is_tech

    def retrieval_tool(self, query: str) -> str:
        """
        Retrieves context for a technical query using the retrieval engine.

        Args:
            query (str): The user's query.

        Returns:
            str: Retrieved context or empty string on error.
        """
        try:
            logger.info("[retrieval_tool] Running for query: %r", query)
            #result = self.rg.generate_response(query)
            data = {"query":query}
            response = requests.post(self.rag_apiendpoint, json=data, timeout=30)
            result = response.json()
            parsed_result = result["context"]
            logger.info("[retrieval_tool] Retrieved context length: %d",
                        len(parsed_result) if parsed_result else 0)
            return parsed_result
        except (AttributeError, RuntimeError, ValueError) as e:
            logger.error("[retrieval_tool] Exception: %s", e)
            logger.debug("%s", traceback.format_exc())
            return ""

    async def llm_with_context(self,
                               messages: List[BaseMessage],
                               context: str,
                               summary: str = "",
                               username: str = "") -> str:
        """
        Generates a response using the LLM with provided context and messages.

        Args:
            messages (List[BaseMessage]): Conversation messages.
            context (str): Technical context.
            summary (str, optional): Conversation summary.
            username (str, optional): Username.

        Returns:
            str: Model response or error message.
        """
        full_conversation_text = "\n".join(
                f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
                for m in messages
            )
        try:
            logger.info("[llm_with_context] Messages length: %d, context length: %d",
                        len(messages), len(context) if context else 0)
            prompt = self._returnprompt(self.prompt_technical)
            formatted_prompt = (await prompt.aformat_prompt(
                technical_docs=context,
                summary=summary,
                messages=full_conversation_text,
                user_name=username
            )).to_messages()
            logger.debug("[llm_with_context] Prompt formatted; invoking model.")
            response = await self.model.ainvoke(formatted_prompt)
            logger.info("[llm_with_context] Received response of length %s",
                        len(response) if isinstance(response, str) else 'N/A')
            return response
        except (AttributeError, RuntimeError, ValueError) as e:
            logger.error("[llm_with_context] Exception: %s", e)
            logger.debug("%s", traceback.format_exc())
            return "Sorry, I encountered an error processing your technical query."

    async def smalltalk_tool(self,
                             messages: List[BaseMessage],
                             summary=None,
                             username:str ="") -> str:
        """
        Handles smalltalk queries using the general prompt.

        Args:
            messages (List[BaseMessage]): Conversation messages.
            summary (str, optional): Conversation summary.
            username (str, optional): Username.

        Returns:
            str: Model response or error message.
        """
        full_conversation_text = "\n".join(
                f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
                for m in messages
            )
        try:
            has_summary = bool((summary or "").strip())
            logger.info("[smalltalk_tool] Messages length: %d, Summary present: %s",
                        len(messages), 'Yes' if has_summary else 'No')
            prompt = self._returnprompt(self.prompt_general)
            if summary:
                formatted_prompt = (await prompt.aformat_prompt(
                    messages=full_conversation_text,
                    summary=summary,
                    user_name=username
                )).to_messages()
            else:
                formatted_prompt = (await prompt.aformat_prompt(
                    messages=full_conversation_text,
                    summary="no summary",
                    user_name=username
                )).to_messages()
            logger.debug("[smalltalk_tool] Prompt formatted; invoking model.")
            response = await self.model.ainvoke(formatted_prompt)
            logger.info("[smalltalk_tool] Received response of length %s",
                        len(response) if isinstance(response, str) else 'N/A')
            return response
        except (AttributeError, RuntimeError, ValueError) as e:
            logger.error("[smalltalk_tool] Exception: %s", e)
            logger.debug("%s", traceback.format_exc())
            return "Sorry, I'm having trouble chatting right now."

    async def create_summary(self,
                       summary_messages: List[BaseMessage]
                       ) -> str:
        """
        Creates a summary from the provided messages.

        Args:
            summary_messages (List[BaseMessage]): Messages to summarize.

        Returns:
            str: Generated summary or error message.
        """
        try:
            prompt = self._returnprompt(self.promt_summary)
            formatted_prompt = (await prompt.aformat_prompt(messages = summary_messages
                )).to_messages()
            logger.info("[create_summary] Called with %d messages", len(summary_messages))
            result = await self.model.ainvoke(formatted_prompt)
            logger.info("[create_summary] Summary generated with length %s",
                        len(result) if isinstance(result, str) else 'N/A')
            return result
        except (AttributeError, RuntimeError, ValueError) as e:
            logger.error("[create_summary] Exception: %s", e)
            logger.debug("%s", traceback.format_exc())
            return "Error generating summary."

    async def routed_response(
        self,
        query: str,
        messages: List[BaseMessage] = None,
        summary: str = "",
        username: str = ""
    ) -> str:
        """
        Routes the query to the appropriate tool (technical or smalltalk).

        Args:
            query (str): The user's query.
            messages (List[BaseMessage], optional): Conversation messages.
            summary (str, optional): Conversation summary.
            username (str, optional): Username.

        Returns:
            str: Model response or error message.
        """
        try:
            logger.info("[routed_response] Routing query: %r", query)
            messages = messages or [HumanMessage(content=query)]

            if self.is_technical(query):
                context = self.retrieval_tool(query)
                if not context.strip():
                    logger.warning("[routed_response] No context found for technical query: %r",
                                   query)
                    return "Sorry, I could not find relevant documents for your question."
                response = await self.llm_with_context(messages,
                                                       context=context,
                                                       summary=summary,
                                                       username=username
                                                       )
                logger.info("[routed_response] Technical response returned")
                return response
            else:
                response = await self.smalltalk_tool(messages,
                                                     summary=summary,
                                                     username=username
                                                     )
                logger.info("[routed_response] Smalltalk response returned")
                return response
        except (AttributeError, RuntimeError, ValueError) as e:
            logger.error("[routed_response] Error: %s", e)
            logger.debug("%s", traceback.format_exc())
            return "Sorry, I encountered an unexpected error while processing your query."
