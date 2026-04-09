from langchain.tools import tool

@tool
def kb_search(query: str) -> str:
    """
    Simula una búsqueda en base de conocimiento interna
    """
    knowledge_base = {
        "langgraph": "LangGraph permite construir flujos de agentes con estado, nodos y edges.",
        "memoria": "La memoria de corto plazo conserva el contexto reciente; la de largo plazo pérsiste hechos útiles.",
        "fastapi": "FastAPI es ideal para exponer agentes como APIs asíncronas y tipadas.",
    }

    query_lower = query.lower()
    for key, value in knowledge_base.items():
        if key in query_lower:
            return value
    return "No encontré información relevante en la base de conocimiento."