from typing import TypedDict, Annotated, List
from langchain.schema import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from apigateway.services.tools import Tools
import asyncio

tl = Tools()
memory = InMemorySaver()

class State(TypedDict):
    query: str
    context: str
    response: str
    messages: Annotated[List[BaseMessage], add_messages]
    summary: str

def start_node(state: State):
    # Initialize or continue message history with role-aware messages
    return {
        "query": state["query"],
        "messages": state.get("messages", []),
        "context": "",
        "summary": "",
        "response": ""
    }

def tool_node(state: State):
    context = tl.retrieval_tool(state["query"])
    return {
        "query": state["query"],
        "context": context,
        "messages": state["messages"],
        "summary": state.get("summary", ""),
        "response": state.get("response", ""),
    }

async def llm_node(state: State):
    # Append current user query as HumanMessage
    messages = state["messages"] + [HumanMessage(content=state["query"])]
    # Generate response string from model (passing messages)
    answer_text = await tl.llm_with_context(messages, context=state["context"])
    # Append bot response as AIMessage
    messages.append(AIMessage(content=answer_text))
    return {
        "response": answer_text,
        "messages": messages,
        "query": state["query"],
        "context": state["context"],
        "summary": state.get("summary", ""),
    }

async def smalltalk_node(state: State):
    messages = state["messages"] + [HumanMessage(content=state["query"])]
    summary = state.get("summary", "")
    summary_message = (
        f"This is summary of the conversation to date: {summary}\n\nExtend the summary by taking into account the new messages above:"
        if summary else None
    )
    answer_text = await tl.smalltalk_tool(messages, summary=summary_message)
    messages.append(AIMessage(content=answer_text))
    return {
        "response": answer_text,
        "messages": messages,
        "summary": summary,
        "query": state["query"],
        "context": state.get("context", ""),
    }

def summarize_conversation(state: State):
    summary = state.get("summary", "")
    if summary:
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    summary_text = tl.create_summary(messages)  # Returns string

    # Prune message history but keep last 2 messages for context
    pruned_messages = state["messages"][-2:] if len(state["messages"]) > 2 else state["messages"]
    return {
        "summary": summary_text,
        "messages": pruned_messages,
        "query": state.get("query", ""),
        "context": state.get("context", ""),
        "response": state.get("response", ""),
    }

def route_by_intent(state: State):
    return "tool" if tl.is_technical(state["query"]) else "smalltalk"

def should_continue(state: State):
    messages = state["messages"]
    query = state.get("query", "").lower().strip()
    stop_words = {"bye", "goodbye", "stop", "exit", "thanks"}
    if query in stop_words or len(messages) > 20:
        return END
    if len(messages) > 6:
        return "summarize_conversation"
    return "smalltalk"

def passthrough_node(state: State):
    return state

# --- Simple chain (no loops): START → start → (tool|smalltalk) → llm/smalltalk_node → summarize_conversation → END

graph = StateGraph(State)
graph.add_node("start", start_node)
graph.add_node("tool", tool_node)
graph.add_node("llm", llm_node)
graph.add_node("smalltalk", smalltalk_node)
graph.add_node("should_continue", passthrough_node)
graph.add_node("summarize_conversation", summarize_conversation)
graph.add_node("end", lambda state: state)

graph.add_edge(START, "start")
graph.add_conditional_edges("start", route_by_intent, ["tool", "smalltalk"])
graph.add_edge("tool", "llm")
graph.add_edge("llm", "should_continue")
graph.add_edge("smalltalk", "should_continue")
graph.add_conditional_edges("should_continue", should_continue, ["smalltalk", "summarize_conversation", END])
graph.add_edge("summarize_conversation", "smalltalk")
graph.add_edge("smalltalk", END)

conversation_graph = graph.compile(checkpointer=memory)
