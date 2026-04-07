from dataclasses import dataclass
from typing import Callable
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelResponse, ModelRequest
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Context:
    user_id: str

@tool
def search_tool(query: str) -> str:
    """Busca información en internet"""
    return f"Resultados de búsqueda para: {query}"

@tool
def analysis_tool(data: str) -> str:
    """Analiza un conjuno de datos complejos"""
    return f"Análisis completado para: {data}"

@tool
def export_tool(data: str) -> str:
    """Exporta los datos a un archivo CSV o PDF"""
    return f"Datos exportados exitosamente: {data}"

@wrap_model_call
def store_based_tools(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Filtra las herramientas basándose en las preferencias/permisos de la Store"""
    context = request.runtime.context
    user_id = context.user_id if context.user_id else None

    available_tools = context.tools or []
    store = request.runtime.store

    #Si no hay user_id o store, no filtrar
    if not user_id or store:
        return handler(request)

    feature_flags = store.get(("features",), user_id)

    if feature_flags:
        enabled_tools = feature_flags.value.get("enabled_tools", [])
        filtered_tools = [t for t in available_tools if t.name in enabled_tools]
        request = request.override(tools=filtered_tools)

    return handler(request)

store = InMemorySaver()

# El usuario "premium_user" tiene acceso a todas las herramientas
store.put(("features",), "premium_user", {
    "enabled_tools": ["search_tool", "analysis_tool", "export_tool"]
})

# El usuario "basic_user" solo tiene acceso a la herramienta de búsqueda
store.put(("features",), "basic_user", {
    "enabled_tools": ["search_tool"]
})

agent = create_agent(
    model=ChatOpenAI(model="gpt-4.1-mini"),
    tools=[search_tool, analysis_tool, export_tool],
    middleware=[store_based_tools],
    context_schema=Context,
    store=store,
)

if __name__ == "__main__":
    prompt = "Busca los datos de ventas de 2023, analizalos y exportalos a PDF"

    print("Ejecutando para basic_user:")
    # Este usuario solo tiene 'search_tool'. El modelo no podrá usar analysis_tool ni export_tool.
    basic_context = Context(user_id="basic_user")
    response_basic = agent.invoke(
        {"messages": [("user", prompt)]},  # type: ignore
        context=basic_context
    )
    print(response_basic["messages"][-1].content)
    print("--------------------------------------------------------------")
    # Este usuario tiene todas las herramientas disponibles
    premium_context = Context(user_id="premium_user")
    response_premium = agent.invoke(
        {"messages": [("user", prompt)]},  # type: ignore
        context=premium_context
    )
    print(response_premium["messages"][-1].content)



