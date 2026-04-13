from typing import Literal

from langgraph.graph import START, END, StateGraph
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime

from context import AppContext
from state import HelixState
from memory import store
from agents import billing_agent, incidents_agent, research_agent
from langchain.messages import HumanMessage

def detect_route(state: HelixState) -> Command[Literal["billing_worker", "incident_worker", "parallel_fanout", "clarify_worker"]]:
    """
    Router explícito. Aquí decides si usas branching determinista
    en vez de delegar todo al LLM.
    """
    msg = state["user_message"].lower()

    billing_hits = any(word in msg for word in ["factura", "billing", "refound", "cobro", "invoice"])
    incident_hits = any(word in msg for word in ["error", "incidente", "caído", "webhook", "latencia", "500"])

    if billing_hits and incident_hits:
        return Command(
            update={
                "route": "mixed",
                "severity": "high",
                "notes": ["Caso mixto detectado: billing + incident."],
                "active_skills": ["billing", "incident"],
            },
            goto="parallel_fanout",
        )

    if billing_hits:
        return Command(
            update={
                "route": "billing",
                "severity": "medium",
                "notes": ["Caso de facturación detectada"],
                "active_skills": ["billing"],
            },
            goto="billing_worker",
        )

    if incident_hits:
        return Command(
            update={
                "route": "incident",
                "severity": "high",
                "notes": ["Caso técnico detectado."],
                "active_skills": ["incident"],
            },
            goto="incident_worker",
        )

    return Command(
        update={
            "route": "clarify",
            "severity": "low",
            "notes": ["No se pudo clasificar con suficiente confianza."],
            "active_skills": ["research"],
        },
        goto="clarify_worker",
    )

def billing_worker(state: HelixState, runtime: Runtime[AppContext]) -> HelixState:
    """Subagente de billing"""
    result = billing_agent.invoke({
        "messages": [HumanMessage(content=(
            "Analiza este caso de billing, usa herramientas si hace falta "
            f"y responde con diagnóstico y siguiente acción:\n\n{state['user_message']}"
        ))],
    }, context=runtime.context)
    content = result["messages"][-1].content
    return {
        "billing_result": str(content),
        "actions_taken": ["billing_agent ejecutado"]
    }

# En detect_route devuelve Command[...] porque ese nodo hace dos cosas a la vez:
# 1. Actualiza estado (update={...}).
# 2. Decide explícitamente el siguiente nodo (goto="billing_worker", "incident_worker", etc.)

# billing_worker devuelve un diccionario de estados porque ese nodo solo está produciendo datos
# (billing_result, actions_taken).
# No decide el flujo; el siguiente paso de los edges del grafo (flujo estático), no del retorno.

# En resumen:
# - Command = update + control de enrutamiento dinámico.
# - dict = solo update de estado

# Además rl tió Command[Literal[...]] en detect_route te da chequeo estático de destinos válidos de goto.

def incident_worker(state: HelixState, runtime: Runtime) -> HelixState:
    """Subagente de incidentes."""
    result = incidents_agent.invoke({
        "messages": [HumanMessage(content=(
            "Analiza este incidente técnico, usa herramientas si hace falta "
            f"y responde con diagnóstico, impacto y siguientes pasos: \n\n{state['user_message']}"
        ))]
    },
    context=runtime.context)

    content = result["messages"][-1].content
    requires_human = runtime.context.environment == "prod"
    return {
        "incident_result": str(content),
        "requires_human_approval": requires_human,
        "actions_taken": ["incident_agent ejecutado"]
    }

def clarify_worker(state: HelixState, runtime: Runtime[AppContext]) -> HelixState:
    """Subagente de investigación /aclaración."""
    result = research_agent.invoke({
        "messages": [HumanMessage(content=(
            "Necesito clasificar mejor el caso. "
            "Propón una breve aclaración para el usuario y posibles hipótesis:\n\n"
            f"{state['user_message']}"
        ))]
    },
    context=runtime.context)
    content = result["messages"][-1].content
    return {
        "research_result": str(content),
        "actions_taken": ["research_agent ejecutado"],
    }

def parallel_fanout(state: HelixState) -> Command[Literal["billing_worker", "incident_worker"]]:
    """
    Hand-off lógico: cuando el caso es mixed, lanzamos primero billing.
    Luego desde el merge decidimos si continuamos o no
    """
    return Command(goto="billing_worker")

def merge_results(state: HelixState) -> Command[Literal["human_approval", "compose_final"]]:
    """Combina resultados parciales y decide si escala a aprobación humana"""
    if state.get("requires_human_approval", False):
        return Command(
            update={
                "escalations": ["Se requiere aprobación humana por acción o contexto de producción."]
            },
            goto="human_approval",
        )
    return Command(goto="compose_final")

def maybe_continue_after_billing(state: HelixState) -> Literal["incident_worker", "merge_results"]:
    """
    Conditional edge clásico: si el caso es mixed, sigue con incident.
    Sí o no, termina el fan-in
    """
    if state.get("route") == "mixed" and "incident_agent ejecutado" not in state.get("actions_taken", []):
        return "incident_worker"
    return "merge_results"

def human_approval(state: HelixState) -> HelixState:
    """
    En una app real, aquí podrías usar interrupt() para aprobar/rechazar.
    Dejamos la marca y seguimos con una salida segura.
    """
    return {
        "notes": ["Aprobación humana pendiente o recomendada antes de acción sensible"],
        "actions_taken": ["human_approval marcado"]
    }

def compose_final(state: HelixState, runtime: Runtime[AppContext]) -> HelixState:
    """
    Nodo final: compone una respuesta ejecutiva y guarda memoria útil.
    """
    parts: list[str] = []

    if state.get("billing_result"):
        parts.append(f"Billing:\n{state['billing_result']}")

    if state.get("incident_result"):
        parts.append(f"Incident:\n{state['incident_result']}")

    if state.get("research_result"):
        parts.append(f"Aclaración sugerida:\n{state['research_result']}")

    if state.get("escalations"):
        parts.append("Escalamiento:\n" + "\n".join(state["escalations"]))

    final_response = "\n\n".join(parts) if parts else "No se obtuvo resultados."

    # Memoria larga: guarda ruta útil del usuario
    if runtime.store is not None:
        runtime.store.put(
            ("last_route_by_user",),
            runtime.context.user_id,
            {
                "route": state.get("route", "clarify"),
            }
        )

    return {
        "final_response": final_response,
        "actions_taken": ["compose_final ejecutado"],
    }

def build_graph():
    builder = StateGraph(HelixState)

    builder.add_node("detect_route", detect_route)
    builder.add_node("billing_worker", billing_worker)
    builder.add_node("incident_worker", incident_worker)
    builder.add_node("clarify_worker", clarify_worker)
    builder.add_node("parallel_fanout", parallel_fanout)
    builder.add_node("merge_results", merge_results)
    builder.add_node("human_approval", human_approval)
    builder.add_node("compose_final", compose_final)

    builder.add_edge(START, "detect_route")

    # Después de billing decidimos si seguimos con incidente o mergeamos
    builder.add_conditional_edges(
        "billing_worker",
        maybe_continue_after_billing,
        {
            "incident_worker": "incident_worker",
            "merge_results": "merge_results",
        },
    )

    builder.add_edge("incident_worker", "merge_results")
    builder.add_edge("clarify_worker", "compose_final")
    builder.add_edge("human_approval", "compose_final")
    builder.add_edge("compose_final", END)

    checkpointer = InMemorySaver()

    return builder.compile(checkpointer=checkpointer, store=store)