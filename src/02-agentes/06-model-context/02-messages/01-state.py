from dataclasses import dataclass
from typing import TypedDict, Callable, List, Annotated
import operator

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from dotenv import load_dotenv

load_dotenv()

# 1. Definir el Contexto (Configuración Estática)
@dataclass
class EnvContext:
    environment: str

# 2. Definir el Estado (Memoria Dinámica)
class AgentState(TypedDict):
    messages: Annotated[List[dict], operator.add]
    acciones_realizadas: int

# 3. Herramientas de ejemplo
@tool
def check_service_status(service_name: str) -> str:
    """Verifica el estado del servicio"""
    return f"El servicio {service_name} esta corriendo"

# 4. Middleware que usa STATE y CONTEXT
@wrap_model_call
def state_and_context_middleware(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:

    # A) Leer el contexto (¿En qué entorno estamos?)
    env = request.runtime.context.environment if request.runtime and request.runtime.context else "prod"

    # B) Leer el Estado (¿Cuántas acciones ha hecho en esta sesión?)
    acciones = request.state.get("acciones_realizadas", 0)

    # Lógica combinada:
    if env == "prod" and acciones >= 3:
        # Si está en producción y ya se realizaron varias acciones
        warning = {
            "role": "system",
            "content": f"ALERTA: Ya has realizado {acciones} en producción"
        }

        request = request.override(messages=[warning, *request.messages])

    return handler(request)

# 5. Crear el agente con AMBOS esquemas
agent = create_agent(
    model="gpt-4o-mini",
    tools = [check_service_status],
    middleware = [state_and_context_middleware],
    context_schema=EnvContext, # Inyectamos el esquema del contexto
    state_schema=AgentState,   # Inyectamos el esquema del estado
)

# 6. Ejecutar
if __name__ == "__main__":
    # Simulamos un estado donde el agente ya hizo 3 acciones previas
    estado_actual = {
        "messages": [{"role": "user", "content": "Quiero realizar más acciones, me recomiendas realizar más acciones?."}],
        "acciones_realizadas": 5,
    }

    # Le pasamos el Estado inicial (como diccionario) y el Contexto (como dataclass)
    response = agent.invoke(
        estado_actual,
        context=EnvContext(environment="prod")
    )

    print(f"🤖 Agente: {response["messages"][-1].content}")


from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver


agent = create_agent(
    model="gpt-4.1",
    tools=[write_file_tool, execute_sql_tool, read_data_tool],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "write_file": True,  # Todas las decisiones (approve, edit, reject) permitidas
                "execute_sql": {"allowed_decisions": ["approve", "reject"]},  # No se permite editar
                # Operación segura, no se necesita aprobación
                "read_data": False,
            },
            # Prefijo para mensajes de interrupción: combinado con el nombre de la herramienta y args para formar el mensaje completo
            # e.g., "Ejecución de la herramienta pendiente de aprobación: execute_sql con query='DELETE FROM...'"
            # Las herramientas individuales pueden anular esto especificando una "description" en su configuración de interrupciones
            description_prefix="Tool execution pending approval",
        ),
    ],
    # Human-in-the-loop requiere checkpointers para gestionar interrupciones
    # In producción, usa un persistente checkpointer como AsyncPostgresSaver.
    checkpointer=InMemorySaver(),
)













