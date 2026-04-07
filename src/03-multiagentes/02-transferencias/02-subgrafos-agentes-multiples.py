from typing import Literal, Annotated
from typing_extensions import TypedDict, NotRequired

from langchain.messages import AIMessage, ToolMessage, HumanMessage
from langchain.tools import tool, InjectedToolCallId, ToolRuntime
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent,AgentState
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command

import os
from dotenv import load_dotenv

load_dotenv()

# =========================================================
# 1) Estado compartido del grafo
# =========================================================
class MultiAgentState(AgentState):
    active_agent: NotRequired[str]

# =========================================================
# 2) Helpers
# =========================================================
def get_last_ai_message(messages) -> AIMessage:
    """Obtiene el último AIMessage del historial."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg
    # Como es transferencia debemos asegurarnos que exista un AIMessage previo.
    raise ValueError("No se encontró ningún AIMessage previo para hacer handoff")

# =========================================================
# 3) Tools de handoff entre agentes
# =========================================================
@tool
def transfer_to_sales(runtime: ToolRuntime[None, MultiAgentState], tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Transfiere al agente de ventas."""
    last_ai_message = get_last_ai_message(runtime.state["messages"])

    transfer_messages = ToolMessage(
        content="Transferido de agente de soporte a agente de ventas",
        tool_call_id=tool_call_id,
    )

    return Command(
        goto="sales_agent",
        update = {
            "active_agent": "sales_agent",
            # importante: pasamos solo el último AIMessage + el ToolMessage
            # Para no inflar contexto innecesariamente
            "messages": [last_ai_message, transfer_messages],
        }
    )

@tool
def transfer_to_support(runtime: ToolRuntime[None, MultiAgentState], tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Transfiere al agente de soporte."""
    last_ai_message = get_last_ai_message(runtime.state["messages"])

    transfer_messages = ToolMessage(
        content="Transferido al agene de soporte desde agente de ventas",
        tool_call_id=tool_call_id,
    )

    return Command(
        goto="support_agent",
        update={
            "active_agent": "support_agent",
            "messages": [last_ai_message, transfer_messages],
        }
    )

# =========================================================
# 4) Modelo moderno
# =========================================================
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# =========================================================
# 5) Agentes especializados
# =========================================================
sales_agent = create_agent(
    model = model,
    tools = [transfer_to_support],
    system_prompt=(
        "Eres un agente de ventas. "
        "Ayuda con precios, planes, compras, renovaciones y cuestiones comerciales. "
        "Si el usuario pregunta sobre problemas técnicos, errores, problemas de inicio de sesión o soporte, "
        "usa la herramienta transfer_to_support"
    ),
    state_schema=MultiAgentState,
)

support_agent = create_agent(
    model = model,
    tools = [transfer_to_sales],
    system_prompt=(
        "Eres una gente de soporte"
        "Ayuda con problemas técnicos, solución de problemas, problemas de inicio de sesión, errores y fallos del sistema. "
        "Si el usuario pregunta sobre precios, planes o compras, "
        "usa la herramienta transfer_to_sales"
    ),
    state_schema=MultiAgentState,
)

# =========================================================
# 6) Nodos del grafo que invocan cada agente
# =========================================================
def call_sales_agent(state: MultiAgentState):
    """Nodo que ejecuta el agente de ventas."""
    return sales_agent.invoke(state)

def call_support_agent(state: MultiAgentState):
    """Nodo que ejecuta el agente de ventas."""
    return support_agent.invoke(state)

# =========================================================
# 7) Routers
# =========================================================
def route_initial(state: MultiAgentState) -> Literal["sales_agent", "support_agent"]:
    """Escoge el agente inicial."""
    return state.get("active_agent") or "sales_agent"

def route_after_agent(
        state: MultiAgentState,
) -> Literal["sales_agent", "support_agent", "__end__"]:
    """
    Si el último mensaje es un AIMessage sin tool_calls, el flujo terminó.
    Si hubo handoff, seguimos con el active_agent.
    """
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
            return "__end__"

    return state.get("active_agent") or "sales_agent"

# =========================================================
# 8) Construcción del grafo padre
# =========================================================
builder = StateGraph(MultiAgentState)

builder.add_node("sales_agent", call_sales_agent)
builder.add_node("support_agent", call_support_agent)

builder.add_conditional_edges(
    START,
    route_initial,
    {
        "sales_agent": "sales_agent",
        "support_agent": "support_agent",
    },
)

builder.add_conditional_edges(
    "sales_agent",
    route_after_agent,
    {
        "sales_agent": "sales_agent",
        "support_agent": "support_agent",
        "__end__": END,
    },
)

builder.add_conditional_edges(
    "support_agent",
    route_after_agent,
    {
        "sales_agent": "sales_agent",
        "support_agent": "support_agent",
        "__end__": END,
    },
)

graph = builder.compile()


# =========================================================
# 9) Ejemplo de uso
# =========================================================
if __name__ == "__main__":
    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Hi, I'm having trouble with my account login. Can you help?",
                }
            ]
        }
    )

    print("\n=== FINAL STATE ===\n")
    print("active_agent:", result.get("active_agent"))
    print()

    for msg in result["messages"]:
        msg.pretty_print()