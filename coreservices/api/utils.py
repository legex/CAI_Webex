
def get_config_with_session(session_id: str) -> dict:
    return {"configurable": {"session_id": session_id, "thread_id": session_id}}