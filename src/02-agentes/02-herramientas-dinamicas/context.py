from dataclasses import dataclass
from typing import Callable
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelResponse, ModelRequest
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Context:
    user_role: int

@tool
def read_data(query: str) -> str:
    """Lee datos de la base de datos de la empresa."""
    return f"Datos leídos exitosamente para la consulta: {query}"

@tool
def write_data(data: str) -> str:
    """Escribe o actualiza nueva información en la base de datos."""
    return f"Nuevos datos registrados: {data}"

@tool
def delete_data(record_id: str) -> str:
    """Elimina permanentemente un registro de la base de datos."""
    return f"Registro {record_id} eliminado del sistema."

@wrap_model_call
def context_based_tools(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    if request.runtime is None or request.runtime.context is None:
        user_role = "viewer"
    else:
        user_role = request.runtime.context.user_role

    tools = request.tools

    if user_role == "admin":
        pass
    elif user_role == "editor":
        tools = [t for t in request.tools if t.name != "delete_data"]
    elif user_role == "viewer":
        tools = [t for t in request.tools if t.name == "read_data"]

    request = request.override(tools=tools)
    return handler(request)

agent = create_agent(
    model="gpt-4.1-mini",  # Ajustado al modelo que prefieras usar
    tools=[read_data, write_data, delete_data],
    system_prompt=(
        "Eres un asistente de base de datos extremadamente estricto. "
        "Sigue estas reglas al pie de la letra:\n"
        "1. REVISA TUS HERRAMIENTAS: Antes de responder, verifica qué herramientas tienes disponibles en esta sesión.\n"
        "2. CERO ALUCINACIONES: NUNCA afirmes haber leído, escrito o eliminado datos a menos que hayas llamado a la herramienta correspondiente y hayas recibido su respuesta.\n"
        "3. PERMISOS DENEGADOS: Si el usuario te pide una acción (como eliminar o escribir) y NO TIENES la herramienta exacta para hacerlo en tu lista actual, DEBES decirle explícitamente: 'Mi rol actual no tiene los permisos necesarios para realizar esta acción' y detenerte ahí para esa tarea en particular.\n"
        "4. EJECUTA LAS HERRAMIENTAS: No inventes los datos del reporte, usa obligatoriamente read_data para obtenerlos antes de hacer el resumen."
    ),
    middleware=[context_based_tools], # type: ignore
    context_schema=Context
)

if __name__ == "__main__":
    # Un prompt ambicioso que intenta usar todas las capacidades
    prompt = "Por favor, lee el reporte de métricas, escribe un resumen con los hallazgos y elimina los datos antiguos del mes pasado."

    print("--- Prueba con ROL: VIEWER ---")
    # El visor (viewer) solo debería poder acceder a 'read_data'
    viewer_context = Context(user_role="viewer")
    response_viewer = agent.invoke(
        {"messages": [("user", prompt)]}, # type: ignore
        context=viewer_context
    )
    print(response_viewer["messages"][-1].content)

    print("\n--- Prueba con ROL: EDITOR ---")
    # El editor debería poder acceder a 'read_data' y 'write_data', pero fallará al intentar 'delete_data'
    editor_context = Context(user_role="editor")
    response_editor = agent.invoke(
        {"messages": [("user", prompt)]}, # type: ignore
        context=editor_context
    )
    print(response_editor["messages"][-1].content)

    print("\n--- Prueba con ROL: ADMIN ---")
    # El admin podrá ejecutar todas las herramientas sin restricciones
    admin_context = Context(user_role="admin")
    response_admin = agent.invoke(
        {"messages": [("user", prompt)]}, # type: ignore
        context=admin_context
    )
    print(response_admin["messages"][-1].content)
