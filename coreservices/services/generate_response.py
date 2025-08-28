import os
import asyncio
from dotenv import load_dotenv
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
from services.langgraphtool import State
from services.langgraph_builder import create_graph

mongouri = os.getenv("MONGO_URI")

async def get_response(state: State, config):
    """functino to generate response
    args:
    state: message state
    config: uid for thread"""
    graph_builder = create_graph()
    try:
        async with AsyncMongoDBSaver.from_conn_string(conn_string=mongouri,
                                                      db_name="chatdata",
                                                      checkpoint_collection_name="checkpoints",
                                                      writes_collection_name="checkpoint_writes",
                                                      ) as checkpointer:
            graph = graph_builder.compile(checkpointer=checkpointer)
            output_generated = await graph.ainvoke(state,config=config)
            return output_generated['response']
    except Exception as e:
        raise RuntimeError(f"Error running conversation workflow: {str(e)}") from e
