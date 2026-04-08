from langchain_classic.agents import Agent
from typing_extensions import Literal, NotRequired, Annotated
from langchain.agents import AgentState, create_agent
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from langchain.tools import tool, ToolRuntime, InjectedToolCallId
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from dotenv import load_dotenv

load_dotenv()

# =========================================================
# 1) Estado compartido
# =========================================================
class HelpdeskState(AgentState):
    active_agent: NotRequired[str]

# =========================================================
# 2) Helper
# =========================================================
def get_last_ai_message(messages) -> AIMessage:
    """Retorna el último AIMessage del historial"""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg
    # Nos aseguramos que exista un AIMessage
    raise ValueError("No se encontró ningún AIMessage previo para hacer handoff")

# =========================================================
# 3) Tools de handoff
# =========================================================
@tool
def transfer_to_billing(
        runtime: ToolRuntime[None, HelpdeskState],
        tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Transfiere la conversación al agente de facturación"""
    last_msg = get_last_ai_message(runtime.state["messages"])

    tools_msg = ToolMessage(
        content="Transferido al agente de facturación desde el agente de preguntas frecuentes",
        tool_call_id=tool_call_id,
    )

    return Command(
        goto="billing_agent",
        update={
            "active_agent": "billing_agent",
            "messages": [last_msg, tools_msg],
        },
        graph=Command.PARENT,
    )

@tool
def transfer_to_faq(
        runtime: ToolRuntime[None, HelpdeskState],
        tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Transfiere la conversación al agente FAQ."""
    last_msg = get_last_ai_message(runtime.state["messages"])

    tools_msg = ToolMessage(
        content="Transferido al agente de preguntas frecuentes desde el agente de facturación",
        tool_call_id=tool_call_id,
    )

    return Command(
        goto="faq_agent",
        update={
            "active_agent": "faq_agent",
            "messages": [last_msg, tools_msg],
        },
        graph=Command.PARENT,
    )

# =========================================================
# 4) Modelo moderno
# =========================================================
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# =========================================================
# 5) Agentes
# =========================================================
faq_agent = create_agent(
    model=model,
    tools=[transfer_to_billing],
    state_schema=HelpdeskState,
    system_prompt=(
        "Eres un agente de preguntas frecuentes. "
        "Responde dudas generales sobre horarios, políticas, uso básico del servicio y ayuda general "
        "Si el usuario pregunta sobre pagos, cobros, facturas, reembolsos o subscripciones, "
        "usa la tool transfer_to_billing."
    )
)

billing_agent = create_agent(
    model = model,
    tools=[transfer_to_faq],
    state_schema=HelpdeskState,
    system_prompt=(
        "Eres un agente de facturación. "
        "Responde dudas sobre pagos, facturas, suscripciones, cargos y reembolsos. "
        "Si el usuario cambia a una duda general no relacionada con facturación, "
        "usa la tool transfer_to_faq."
    )
)

# =========================================================
# 6) Nodos del grafo padre
# =========================================================
def call_faq_agent(state: HelpdeskState):
    return faq_agent.invoke(state)

def call_billing_agent(state: HelpdeskState):
    return billing_agent.invoke(state)

# =========================================================
# 7) Routing del grafo
# =========================================================
def route_initial(state: HelpdeskState) -> Literal["faq_agent", "billing_agent"]:
    return state.get("active_agent") or "billing_agent"

def route_after_agent(state: HelpdeskState) -> Literal["faq_agent", "billing_agent", "__end__"]:
    messages = state.get("messages", [])

    if messages:
        last_msg = messages[-1]

        # Si el último mensaje es IA y no tiene tool_calls, terminó
        if isinstance(last_msg, AIMessage):
            return "__end__"

    # Si hubo transferencia seguimos con el agente activo
    return state.get("active_agent") or "faq_agent"

# =========================================================
# 8) Construcción del grafo padre
# =========================================================
builder = StateGraph(HelpdeskState)

builder.add_node("faq_agent", call_faq_agent)
builder.add_node("billing_agent", call_billing_agent)

builder.add_conditional_edges(
    START,
    route_initial,
    {
        "faq_agent": "billing_agent",
        "billing_agent": "faq_agent",
    }
)

builder.add_conditional_edges(
    START,
    route_after_agent,
    {
        "faq_agent": "billing_agent",
        "billing_agent": "faq_agent",
        "__end__": END,
    }
)

graph = builder.compile()

# =========================================================
# 9) Prueba
# =========================================================
if __name__ == "__main__":
    result = graph.invoke(HumanMessage(content="Me llegó un cobro duplicado, ¿pueden revisar mi factura?"))
    print("Agente activo final:", result.get("active_agent"))
    print()

    for msg in result["messages"]:
        msg.pretty_print()