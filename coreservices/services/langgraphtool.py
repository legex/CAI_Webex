import re
from typing import TypedDict, Annotated, List
from langchain.schema import HumanMessage, AIMessage, BaseMessage
from langchain_core.messages import RemoveMessage
from langgraph.graph.message import add_messages
from langgraph.graph import END
from coreservices.services.tools import Tools
from coreservices.services.websearch import WebSearch
from datamanagement.core.logger import setup_logger

logger = setup_logger("nodelog", 'datamanagement/log/nodes.log')

web_search = WebSearch()

tl = Tools()

class State(TypedDict):
    """
    State dictionary for conversation flow in the graph.
    """
    query: str
    context: str
    response: str
    messages: Annotated[List[BaseMessage], add_messages]
    summary: str
    user_name: str

def start_node(state: State):
    """
    Initializes or continues the message history with role-aware messages.

    Args:
        state (State): The current state.

    Returns:
        dict: Updated state with initialized fields.
    """
    logger.info("start_node called with query: %s", state.get('query', ''))
    result = {
        "query": state["query"],
        "messages": state.get("messages", []),
        "context": "",
        "summary": state.get("summary", ""),
        "response": "",
        "user_name": state.get("user_name", "")
    }
    logger.debug("start_node result: %s", result)
    return result

def tool_node(state: State):
    """
    Retrieves context for the query using the retrieval tool.

    Args:
        state (State): The current state.

    Returns:
        dict: Updated state with retrieved context.
    """
    logger.info("tool_node called with query: %s", state.get('query', ''))
    context_web = web_search.modelcall(state["query"])
    context_vectorsearch = tl.retrieval_tool(state["query"])
    context = context_vectorsearch+context_web
    logger.debug("tool_node retrieved context: %s", context)
    result = {
        "query": state["query"],
        "context": context,
        "messages": state["messages"],
        "summary": state.get("summary", ""),
        "response": state.get("response", ""),
    }
    logger.debug("tool_node result: %s", result)
    return result

def extract_name_node(state: State):
    """
    Extracts the user's name from the query if present.

    Args:
        state (State): The current state.

    Returns:
        dict: Updated state with extracted user name if found.
    """
    logger.info("extract_name_node called with query: %s", state.get('query', ''))
    query = state["query"]
    current_name = state.get("user_name", "")

    # Basic regex to extract name after "I am", "my name is" etc.
    pattern = r"(?:i am|my name is)\s+([A-Za-z]+)"
    match = re.search(pattern, query, re.IGNORECASE)
    if match:
        extracted_name = match.group(1).strip()
        logger.info("extract_name_node extracted name: %s", extracted_name)
        if extracted_name.lower() != current_name.lower():
            result = {**state, "user_name": extracted_name}
            logger.debug("extract_name_node result: %s", result)
            return result
    logger.debug("extract_name_node no name extracted or name unchanged.")
    return state

async def rag_invoke_node(state: State):
    """
    Generates a technical response using the LLM with context.

    Args:
        state (State): The current state.

    Returns:
        dict: Updated state with response and messages.
    """
    logger.info("rag_invoke_node called with query: %s", state.get('query', ''))
    messages = state["messages"] + [HumanMessage(content=state["query"])]
    logger.debug("rag_invoke_node messages: %s", messages)
    answer_text = await tl.llm_with_context(messages,
                                            context=state["context"],
                                            summary=state.get("summary", ""),
                                            username=state.get("user_name", "")
                                            )
    logger.info("rag_invoke_node LLM answer: %s", answer_text)
    messages.append(AIMessage(content=answer_text))
    result = {
        "response": answer_text,
        "messages": messages,
        "query": state["query"],
        "context": state["context"],
        "summary": state.get("summary", ""),
        "user_name": state.get("user_name", ""),
    }
    logger.debug("rag_invoke_node result: %s", result)
    return result

async def smalltalk_node(state: State):
    """
    Handles smalltalk queries and generates a response.

    Args:
        state (State): The current state.

    Returns:
        dict: Updated state with response and messages.
    """
    logger.info("smalltalk_node called with query: %s", state.get('query', ''))
    messages = state["messages"] + [HumanMessage(content=state["query"])]
    summary = state.get("summary") or ""
    summary_message = (f"This is summary of the conversation to date: {summary}\n\nExtend the summary by taking into account the new messages above:" if summary else None)
    logger.debug("smalltalk_node messages: %s, summary_message: %s", messages, summary_message)
    answer_text = await tl.smalltalk_tool(messages,
                                          summary=summary_message,
                                          username=state.get("user_name", "")
                                          )
    logger.info("smalltalk_node answer: %s", answer_text)
    messages.append(AIMessage(content=answer_text))
    result = {
        "response": answer_text,
        "messages": messages,
        "summary": summary,
        "query": state["query"],
        "context": state.get("context", ""),
        "user_name": state.get("user_name", ""),
    }
    logger.debug("smalltalk_node result: %s", result)
    return result

async def summarize_conversation(state: State):
    """
    Summarizes the conversation based on the message history.

    Args:
        state (State): The current state.

    Returns:
        dict: Updated state with summary and pruned messages.
    """
    logger.info("summarize_conversation called")
    summary = state.get("summary", "")
    if summary:
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    logger.debug("summarize_conversation messages: %s", messages)
    summary_text = await tl.create_summary(messages)
    logger.info("summarize_conversation summary_text: %s", summary_text)

    pruned_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    logger.debug("summarize_conversation pruned_messages: %s", pruned_messages)
    result = {
        "summary": summary_text,
        "messages": pruned_messages,
        "query": state.get("query", ""),
        "context": state.get("context", ""),
        "response": state.get("response", ""),
        "user_name": state.get("user_name", "")
    }
    logger.debug("summarize_conversation result: %s", result)
    return result

def route_by_intent(state: State):
    """
    Determines the next node based on query intent (technical or smalltalk).

    Args:
        state (State): The current state.

    Returns:
        str: The next node name ("tool" or "smalltalk").
    """
    logger.info("route_by_intent called with query: %s", state.get('query', ''))
    intent = "tool" if tl.is_technical(state["query"]) else "smalltalk"
    logger.info("route_by_intent determined intent: %s", intent)
    return intent

def should_summarize(state: State):
    """Node to determine whether to summarize or not"""
    logger.info("should_summarize called")
    messages = state["messages"]
    logger.debug("should_summarize message count: %s", len(messages))
    if len(messages) > 6:
        logger.info("should_summarize: will summarize_conversation")
        return "summarize_conversation"
    logger.info("should_summarize: will END")
    return END

async def connector_node(state: State): # pylint: disable=unused-argument
    """Passthrough state"""
    logger.info("connector_node called")
    return {}
