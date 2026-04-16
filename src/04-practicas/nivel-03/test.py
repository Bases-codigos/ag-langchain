import json
from typing_extensions import Literal, Optional, TypedDict

from langchain.tools import tool
from langchain.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from dotenv import load_dotenv

load_dotenv()

# =========================================================
# 1) Tools reales/simuladas
# =========================================================
FAKE_INCIDENT_DB = {
    "INC-100": {
        "status": "OPEN",
        "priority": "HIGH",
        "hours_open": 3,
        "assigned_to": "team-network",
        "summary": "Pérdida intermitente de conectividad"
    },
    "INC-200": {
        "status": "RESOLVED",
        "priority": "MEDIUM",
        "hours_open": 1,
        "assigned_to": "team-app",
        "summary": "Error temporal en autenticación"
    },
    "INC-300": {
        "status": "OPEN",
        "priority": "HIGH",
        "hours_open": 52,
        "assigned_to": None,
        "summary": "Servicio caído y sin responsable asignado"
    },
}

@tool
def get_incident(incident_id: str) -> str:
    """Obtiene el detalle de un incidente por ID."""
    data = FAKE_INCIDENT_DB.get(incident_id)
    if not data:
        return json.dumps(
            {"found": False, "incident_id": incident_id, "error": "Incidente no encontrado"},
            ensure_ascii=False
        )
    return json.dumps(
        {"found": True, "incident_id": incident_id, **data}, ensure_ascii=False
    )

@tool
def get_escalation_policy(priority: str, hours_open: int, assigned_to: Optional[str]) -> str:
    """Evalúa si un incidente debería escalarse según una política sencilla."""
    should_escalate = (
        priority == "HIGH" and hours_open >= 4
    ) or (assigned_to is None) or (hours_open >= 24)

    reason = []

    if priority == "HIGH" and hours_open >= 4:
        reason.append("prioridad alta con demasiadas horas abierto")
    if assigned_to is None:
        reason.append("no tiene responsable asignado")
    if hours_open >= 24:
        reason.append("supera 24 horas abierto")

    return json.dumps(
        {
            "should_escalate": should_escalate,
            "reasons": reason or ["no cumple criterios de escalamiento"]
        }
    )

TOOLS = {
    "get_incident": get_incident,
    "get_escalation_policy": get_escalation_policy,
}

# =========================================================
# 2) Modelos estructurados para planner / critic / replanner
# =========================================================
class PlanStep(TypedDict):
    description: str
    tool_name: Optional[str]
    tool_args: dict
    status: Literal["pending", "done", "failed"]

class PlannerOutput(TypedDict):
    steps: list[PlanStep]

class CriticOutput(TypedDict):
    is_step_sufficient: bool
    needs_replan: bool
    critique: str

class ReplannerOutput(TypedDict):
    steps: list[PlanStep]
    rationale: str

# =========================================================
# 3) Estado del grafo
# =========================================================
class AgentState(TypedDict, total=False):
    user_goal: str
    incident_id: str

    plan: list[PlanStep]
    current_step_index: int

    last_tool_result: str
    critic_result: str
    needs_replan: bool

    collected_evidence: list[dict]
    final_answer: str

    iteration_count: int
    max_iterations: int
    finished: bool

# =========================================================
# 4) LLM
# =========================================================

# Ajusta el modelo según tu proveedor/configuración
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

planner_llm = llm.with_structured_output(PlannerOutput)
critic_llm = llm.with_structured_output(CriticOutput)
replanner_llm = llm.with_structured_output(ReplannerOutput)

# =========================================================
# 5) Nodos
# =========================================================
def planner(state: AgentState) -> AgentState:
    prompt = [
        SystemMessage(
            content=(
                "Eres un planner de un agente de soporte técnico. "
                "Debes generar un plan mínimo y útil. "
                "Solo crea pasos accionables. "
                "Cada paso puede usar una tool concreta o ser parte de un paso final sin tool."
                "Tools disponibles:\n"
                "- get_incident(incident_id)\n"
                "- get_escalation_policy(priority, hours_open, assigned_to)\n"
                "No inventes tools."
            )
        ),
        HumanMessage(
            content=(
                f"Objetivo del usuario: {state['user_goal']}\n"""
                f"ID de incidente: {state['incident_id']}\n\n"
                f"Genera un plan corto para investigar el incidente y decidir si debe escalarse."
            )
        )
    ]

    result = planner_llm.invoke(prompt)

    return {
        "plan": result["steps"],
        "current_step_index": 0,
        "max_iterations": state.get("max_iterations", 8),
        "collected_evidence": [],
        "needs_replan": False,
        "finished": False,

    }

def _is_placeholder(value: object) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.lower()
    return "obtenido del paso anterior" in normalized

def _extract_incident_data(last_tool_result: str) -> Optional[dict]:
    try:
        parsed = json.loads(last_tool_result)
    except (TypeError, json.JSONDecodeError):
        return None

    if not isinstance(parsed, dict):
        return None

    required_fields = {"priority", "hours_open", "assigned_to"}
    if not required_fields.issubset(parsed):
        return None

    return parsed

def _resolve_tool_args(tool_name: Optional[str], tool_args: dict, state: AgentState) -> dict:
    if tool_name != "get_escalation_policy":
        return tool_args

    resolved_args = dict(tool_args or {})
    incident_data = _extract_incident_data(state.get("last_tool_result", ""))
    if not incident_data:
        return resolved_args

    for key in ("priority", "hours_open", "assigned_to"):
        current_value = resolved_args.get(key)
        if key not in resolved_args or _is_placeholder(current_value):
            resolved_args[key] = incident_data.get(key)

    return resolved_args

def executor(state: AgentState) -> AgentState:
    idx = state["current_step_index"]
    plan = state["plan"]

    if idx >= len(plan):
        return {
            "last_tool_result": "No hay más pasos por ejecutar",
        }

    step = plan[idx]

    # Si no requiere tool, tratamos el paso como lógico/textual
    if not step.get("tool_name"):
        result = f"Paso lógico completado: {step['description']}"
        updated_plan = plan[:]
        updated_plan[idx]["status"] = "done"
        evidence = state.get("collected_evidence", []) + [result]
        return {
            "last_tool_result": result,
            "plan": updated_plan,
            "collected_evidence": evidence,
        }

    tool_name = step.get("tool_name")
    tool_args = _resolve_tool_args(tool_name, step.get("tool_args", {}), state)

    tool_fn = TOOLS.get(tool_name)
    if not tool_fn:
        result = json.dumps(
            {"error": f"Tool no encontrada: {tool_name}"},
            ensure_ascii=False,
        )
        updated_plan = plan[:]
        updated_plan[idx]["status"] = "failed"
        evidence = state.get("collected_evidence", []) + [result]
        return {
            "last_tool_result": result,
            "plan": updated_plan,
            "collected_evidence": evidence,
        }
    try:
        tool_result = tool_fn.invoke(tool_args)
        updated_plan = plan[:]
        updated_plan[idx]["status"] = "done"
        evidence = state.get("collected_evidence", []) + [
            f"Paso: {step['description']}\nResultado: {tool_result}"
        ]
        return {
            "last_tool_result": tool_result,
            "plan": updated_plan,
            "collected_evidence": evidence,
        }
    except Exception as e:
        updated_plan = plan[:]
        updated_plan[idx]["status"] = "failed"
        error_result = json.dumps({"error": str(e)}, ensure_ascii=False)
        evidence = state.get("collected_evidence", []) + [error_result]
        return {
            "last_tool_result": error_result,
            "plan": updated_plan,
            "collected_evidence": evidence,
        }

def critic(state: AgentState) -> AgentState:
    idx = state["current_step_index"]
    step = state["plan"][idx]

    prompt = [
        SystemMessage(
            content=(
                "Eres un critic de un agente. "
                "Debes revisar si el resultado del paso actual fue sofisticado. "
                "Marca needs_replan=true si el plan debe ajustarse. "
                "Marca is_step_sufficient=false si el resultado del paso no sirve."
            )
        ),
        HumanMessage(
            content=(
                f"Objetivo: {state['user_goal']}\n"
                f"Paso actual: {json.dumps(step, ensure_ascii=False)}\n"
                f"Resultado del paso: {state['last_tool_result']}\n"
                f"Evidence acumulada: {json.dumps(state.get('collected_evidence', []), ensure_ascii=False)}"
            )
        )
    ]

    result = critic_llm.invoke(prompt)

    return {
        "critic_result": result["critique"],
        "needs_replan": result["needs_replan"]
    }

def replanner(state: AgentState) -> AgentState:
    prompt = [
        SystemMessage(
            content=(
                "Eres un replanner. "
                "Debes corregir el plan actual si el critic detectó fallas. "
                "Conserva pasos válidos y corrige solo lo necesario. "
                "No inventes tools nuevas. "
                "Tools disponibles:\n"
                "- get_incident(incident_id)\n"
                "- get_escalation_policy(priority, hours_open, assigned_to)"
            )
        ),
        HumanMessage(
            content=(
                f"Objetivo: {state['user_goal']}\n"
                f"Plan actual: {json.dumps(state['plan'], ensure_ascii=False)}\n"
                f"Paso actual índice: {state['current_step_index']}\n"
                f"Crítica: {state['critic_result']}\n"
                f"Último resultado: {state['last_tool_result']}\n"
                f"Evidencia acumulada: {json.dumps(state.get('collected_evidence', []), ensure_ascii=False)}"
            )
        )
    ]

    result = replanner_llm.invoke(prompt)

    return {
        "plan": result["steps"],
        "current_step_index": 0,
        "needs_replan": False,
    }

def advance_or_finish(state: AgentState) -> AgentState:
    idx = state["current_step_index"]
    plan = state["plan"]
    iteration_count = state.get("iteration_count", 0) + 1
    max_iterations = state.get("max_iterations", 8)

    # Evitar loops infinitos
    if iteration_count >= max_iterations:
        return {
            "iteration_count": iteration_count,
            "finished": True,
        }

    if idx >= len(plan) - 1:
        return {
            "iteration_count": iteration_count,
            "finished": True,
        }

    return {
        "iteration_count": iteration_count,
        "current_step_index": idx + 1,
    }

def responder(state: AgentState) -> AgentState:
    prompt = [
        SystemMessage(
            content=(
                "Eres un analista técnico. "
                "Debes responder al usuario de forma clara y breve. "
                "Usa solo la evidencia disponible. "
                "No inventes datos."
            )
        ),
        HumanMessage(
            content=(
                f"Objetivo original: {state['user_goal']}\n"
                f"Evidencia recopilada:\n{json.dumps(state.get('collected_evidence', []), ensure_ascii=False, indent=2)}\n\n"
                "Redacta una respuesta final indicando si debe escalarse y por qué."
            )
        )
    ]

    answer = llm.invoke(prompt)

    return {
        "final_answer": answer.content,
        "finished": True,
    }

# =========================================================
# 6) Ruteo
# =========================================================
def after_critic(state: AgentState) -> Literal["replanner", "advance_or_finish"]:
    if state.get("needs_replan", False):
        return "replanner"
    return "advance_or_finish"

def after_advance(state: AgentState) -> Literal["executor", "responder"]:
    if state.get("finished", False):
        return "responder"
    return "executor"

# =========================================================
# 7) Grafo
# =========================================================
builder = StateGraph(AgentState)

builder.add_node("planner", planner)
builder.add_node("executor", executor)
builder.add_node("critic", critic)
builder.add_node("replanner", replanner)
builder.add_node("advance_or_finish", advance_or_finish)
builder.add_node("responder", responder)

builder.add_edge(START, "planner")
builder.add_edge("planner", "executor")
builder.add_edge("executor", "critic")

builder.add_conditional_edges(
    "critic",
    after_critic,
    {
        "replanner": "replanner",
        "advance_or_finish": "advance_or_finish",
    }
)

builder.add_edge("replanner", "executor")
builder.add_conditional_edges(
    "advance_or_finish",
    after_advance,
    {
        "executor": "executor",
        "responder": "responder",
    }
)

builder.add_edge("responder", END)
graph = builder.compile()

# =========================================================
# 8) Ejecución
# =========================================================
if __name__ == "__main__":
    initial_state: AgentState = {
        "user_goal": "Revisa el incidente INC-300 y dime si debo escalarlo",
        "incident_id": "INC-300",
        "max_iterations": 8,
    }

    result = graph.invoke(initial_state)

    print("\n=== RESPUESTA FINAL ===")
    print(result["final_answer"])

    print("\n=== EVIDENCIA ===")
    for e in result.get("collected_evidence", []):
        print("-", e)
