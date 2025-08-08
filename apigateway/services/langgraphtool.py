import re
from typing import TypedDict, Annotated, List
from langchain.schema import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from apigateway.services.tools import Tools


tl = Tools()
memory = InMemorySaver()

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
    return {
        "query": state["query"],
        "messages": state.get("messages", []),
        "context": "",
        "summary": "",
        "response": "",
        "user_name": state.get("user_name", "")
    }

def tool_node(state: State):
    """
    Retrieves context for the query using the retrieval tool.

    Args:
        state (State): The current state.

    Returns:
        dict: Updated state with retrieved context.
    """
    context = tl.retrieval_tool(state["query"])
    return {
        "query": state["query"],
        "context": context,
        "messages": state["messages"],
        "summary": state.get("summary", ""),
        "response": state.get("response", ""),
    }

def extract_name_node(state: State):
    """
    Extracts the user's name from the query if present.

    Args:
        state (State): The current state.

    Returns:
        dict: Updated state with extracted user name if found.
    """
    query = state["query"]
    current_name = state.get("user_name", "")

    # Basic regex to extract name after "I am", "my name is" etc.
    pattern = r"(?:i am|my name is)\s+([A-Za-z]+)"
    match = re.search(pattern, query, re.IGNORECASE)
    if match:
        extracted_name = match.group(1).strip()
        if extracted_name.lower() != current_name.lower():
            return {**state, "user_name": extracted_name}
    return state

async def llm_node(state: State):
    """
    Generates a technical response using the LLM with context.

    Args:
        state (State): The current state.

    Returns:
        dict: Updated state with response and messages.
    """
    messages = state["messages"] + [HumanMessage(content=state["query"])]
    answer_text = await tl.llm_with_context(messages,
                                            context=state["context"],
                                            summary=state.get("summary", ""),
                                            username=state.get("user_name", "")
                                            )
    messages.append(AIMessage(content=answer_text))
    return {
        "response": answer_text,
        "messages": messages,
        "query": state["query"],
        "context": state["context"],
        "summary": state.get("summary", ""),
        "user_name": state.get("user_name", ""),
    }


async def smalltalk_node(state: State):
    """
    Handles smalltalk queries and generates a response.

    Args:
        state (State): The current state.

    Returns:
        dict: Updated state with response and messages.
    """
    messages = state["messages"] + [HumanMessage(content=state["query"])]
    summary = state.get("summary", "")
    summary_message = (f"This is summary of the conversation to date: {summary}\n\nExtend the summary by taking into account the new messages above:" if summary else None)
    answer_text = await tl.smalltalk_tool(messages,
                                          summary=summary_message,
                                          username=state.get("user_name", "")
                                          )
    messages.append(AIMessage(content=answer_text))
    return {
        "response": answer_text,
        "messages": messages,
        "summary": summary,
        "query": state["query"],
        "context": state.get("context", ""),
        "user_name": state.get("user_name", ""),
    }


def summarize_conversation(state: State):
    """
    Summarizes the conversation based on the message history.

    Args:
        state (State): The current state.

    Returns:
        dict: Updated state with summary and pruned messages.
    """
    summary = state.get("summary", "")
    if summary:
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    summary_text = tl.create_summary(messages) 

    pruned_messages = state["messages"][-2:] if len(state["messages"]) > 2 else state["messages"]
    return {
        "summary": summary_text,
        "messages": pruned_messages,
        "query": state.get("query", ""),
        "context": state.get("context", ""),
        "response": state.get("response", ""),
        "user_name": state.get("user_name", "")
    }

def route_by_intent(state: State):
    """
    Determines the next node based on query intent (technical or smalltalk).

    Args:
        state (State): The current state.

    Returns:
        str: The next node name ("tool" or "smalltalk").
    """
    return "tool" if tl.is_technical(state["query"]) else "smalltalk"


graph = StateGraph(State)
graph.add_node("start", start_node)
graph.add_node("extract_name", extract_name_node)
graph.add_node("tool", tool_node)
graph.add_node("llm", llm_node)
graph.add_node("smalltalk", smalltalk_node)
graph.add_node("summarize_conversation", summarize_conversation)
graph.add_node("end", lambda state: state)

graph.add_edge(START, "start")
graph.add_edge("start", "extract_name")
graph.add_conditional_edges("extract_name", route_by_intent, ["tool", "smalltalk"])
graph.add_edge("tool", "llm")
graph.add_edge("llm", END)
graph.add_edge("smalltalk", END)
graph.add_edge("summarize_conversation", END)

conversation_graph = graph.compile(checkpointer=memory)
