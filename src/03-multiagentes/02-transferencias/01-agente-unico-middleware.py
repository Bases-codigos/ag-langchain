from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.tools import tool, ToolRuntime, InjectedToolCallId
from langchain.messages import ToolMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from typing import Callable, Optional, Annotated
import os
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

@tool
def provide_solution(problema: str) -> str:
    """Proporciona una solución técnica basada en la garantía"""
    return f"Solución aplicada para el problema reportado: {problema}"

@tool

def escalate() -> str:
    """Escala a un agente humano si la solución automática no basta"""
    return "El ticket ha sido escalado a un especialista humano de Nivel 2."

# 1. Define state con el rastreador current_step
class SupportState(AgentState):
    """Realizar un seguimiento de qué paso está activo actualmente."""
    current_step: str
    warranty_status: Optional[str]
    
# 2. Las herramientas actualizan current_step via Command
@tool
def record_warranty_status(
    status: str,
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Registre el estado de la garantía y pase al siguiente paso."""
    return Command(update={
        "messages": [
            ToolMessage(
                content=f"Estado de garantía registrado: {status}",
                tool_call_id=tool_call_id
            )
        ],
        "warranty_status": status,
        "current_step": "specialist"
    })
    
# 3. El middleware aplica una configuración dinámica basada en current_step.
@wrap_model_call
def apply_step_config(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Configurar el comportamiento dela gente basado en current_step."""
    step = request.state.get("current_step", "triage")
    
    # Mapear los pasos a sus configuraciones
    configs = {
        "triage": {
            "prompt": "Eres el agente de triage. Tu único trabajo es recopilar información de garantía usando la herramienta record_warranty_status.",
            "tools": [record_warranty_status]
        },
        "specialist": {
            "prompt": "Eres el especialista. Proporciona soluciones basadas en garantía: {warranty_status}",
            "tools": [provide_solution, escalate]
        }
    }
    
    config = configs[step]
    
    # Manejamos el formato de strings en forma segura
    estado_actual = request.state.copy()
    if estado_actual.get("warranty_status") is None:
        estado_actual["warranty_status"] = "Desconocida"
        
    request = request.override(
        system_prompt=config["prompt"].format(**estado_actual),
        tools=config["tools"]
    )
    return handler(request)

# 4. Create agent with middleware
agent = create_agent(
    model,
    tools=[record_warranty_status, provide_solution, escalate],
    state_schema=SupportState,
    middleware=[apply_step_config],
    checkpointer=InMemorySaver()  # Persist state across turns  #
)

# --- EJECUCIÓN DEL AGENTE ---

print("=== INICIANDO SISTEMA DE SOPORTE ===")
configuracion_hilo = {"configurable": {"thread_id": "soporte_usuario_99"}}

print("\n--- TURNO 1: Triage ---")
# El usuario inicia la conversación. El agente arranca en modo "triage".
respuesta_triage = agent.invoke(
    {"messages": [HumanMessage(content="Hola, mi pantalla parpadea. Mi garantía está 'Activa'.")]},
    config=configuracion_hilo
)
print(respuesta_triage["messages"][-1].content)


print("\n--- TURNO 2: Especialista ---")
# El usuario hace una pregunta de seguimiento. El agente ya mutó a "specialist" gracias al middleware.
respuesta_especialista = agent.invoke(
    {"messages": [HumanMessage(content="¿Entonces qué procede? ¿Me lo arreglan?")]},
    config=configuracion_hilo
)
print(respuesta_especialista["messages"][-1].content)