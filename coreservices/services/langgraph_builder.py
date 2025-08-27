from functools import lru_cache
from langgraph.graph import StateGraph, START, END
from coreservices.services.langgraphtool import (
    State,
    smalltalk_node,
    summarize_conversation,
    route_by_intent,
    rag_invoke_node,
    extract_name_node,
    start_node,
    tool_node,
    should_summarize,
    connector_node
    )

@lru_cache(maxsize=2)
def create_graph():
    graph_builder = StateGraph(State)
    graph_builder.add_node("start", start_node)
    graph_builder.add_node("extract_name", extract_name_node)
    graph_builder.add_node("tool", tool_node)
    graph_builder.add_node("rag_invoke_node", rag_invoke_node)
    graph_builder.add_node("smalltalk", smalltalk_node)
    graph_builder.add_node("summarize_conversation", summarize_conversation)
    graph_builder.add_node("connector_node", connector_node)
    graph_builder.add_node("end", lambda state: state)

    graph_builder.add_edge(START, "start")
    graph_builder.add_edge("start", "extract_name")
    graph_builder.add_conditional_edges("extract_name", route_by_intent, ["tool", "smalltalk"])
    graph_builder.add_edge("tool", "rag_invoke_node")
    graph_builder.add_edge("rag_invoke_node", "connector_node")
    graph_builder.add_edge("smalltalk", "connector_node")
    graph_builder.add_conditional_edges("connector_node", should_summarize, ['summarize_conversation', END])
    graph_builder.add_edge("summarize_conversation", END)

    return graph_builder


conversation_graph = create_graph().compile()