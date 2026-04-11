from dataclasses import dataclass
from typing import Callable, Any, Literal, TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, wrap_model_call, ModelRequest, ModelResponse
from langchain_openai import ChatOpenAI
from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage, AIMessage
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 1) CONTEXTO Y ESTADO DE RUTA
# ============================================================
Route = Literal["operations", "docs", "architecture", "general"]

@dataclass
class AppContext:
    user_id: str
    team_id: str
    env: Literal["dev", "qa", "prod"]

class RequestState(TypedDict, total=False):
    route: Route
    skill_name: str

checkpointer = InMemorySaver()
store = InMemoryStore()

# ============================================================
# 2) MEMORIA LARGA
# ============================================================
store.put(
    ("preferences",),
    "user-123",
    {
        "style": "short",
        "audience": "engineering",
    },
)

store.put(
    ("skills",),
    "operations_skill",
    {
        "name": "operations_skill",
        "instructions": (
            "Te enfocas en troubleshooting operativo. "
            "Debes priorizar pasos accionables, validaciones rápidas, "
            "riesgo y siguiente acción concreta"
        ),
        "few_shorts": [
            {
                "input": "payments-api devuelve 502 tras despliegues",
                "output": "Hipótesis: upstream inestable, timeout, pool saturado."
                            "Acción inmediata: revisar salud upstream y despliegue reciente"
            }
        ],
    },
)

store.put(
    ("skills",),
    "docs_skill",
    {
        "name": "docs_skill",
        "instructions": (
            "Te enfocas en documentación interna. "
            "Resume, compara y reorganiza información sin inventar. "
            "Si falta contexto dilo."
        ),
        "few_shorts": [
            "input: resume esta política",
            "output: Objetivo, reglas clave, excepciones y próximos pasos."
        ],
    },
)

store.put(
    ("skills",),
    "architecture_skill",
    {
        "name": "architecture_skill",
        "instructions": (
            "Te enfocas en el diseño de sistemas con LangChain/LangGraph. "
            "Explica trade-offs, patrones, fallos comunes y criterios de elección"
        ),
        "few_shorts": [
            {
                "input": "subagentes vs handoffs",
                "output": "Subagentes para control centralizado; handoffs cuando cambia el dueño de la conversación."
            }
        ],
    },
)

store.put(
    ("runbooks",),
    "rb_502",
    {
        "title": "HTTP 502 troubleshooting",
        "steps": [
            "Verificar upstream",
            "Correlacionar con despliegues recientes",
            "Revisar timeouts y saturación",
            "Confirmar errores 5xx en servicio dependiente."
        ],
    },
)

store.put(
    ("docs",),
    "deploy_policy",
    {
        "name": "Deployment policy",
        "content": (
            "Todos los despliegues a producción requieren validación automática, "
            "rollback documentado y ventana aprobada para servicios críticos."
        ),
    },
)

# ============================================================
# 3) MODELOS
# ============================================================
fast_model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
deep_model = ChatOpenAI(model="gpt-4.1", temperature=0)

# ============================================================
# 4) ROUTER DETERMINISTA
# ============================================================
def route_request(user_text: str) -> Route:
    text = user_text.lower()

    if any(word in text for word in ["runbook", "502", "latencia", "incidente", "troubleshooting"]):
        return "operations"
    if any(word in text for word in ["documento", "política", "resume", "manual", "guía"]):
        return "docs"
    if any(word in text for word in ["arquitectura", "langgraph", "subagentes", "handoff", "router", "skill"]):
        return "architecture"
    return "general"

def skill_for_route(route: Route) -> str | None:
    mapping: dict[Route, str|None] = {
        "operations": "operations_skill",
        "docs": "docs_skill",
        "architecture": "architecture_skill",
        "general": None,
    }
    return mapping.get(route, None)

# ============================================================
# 5) TOOLS
# ============================================================
@tool
def get_user_preferences(runtime: ToolRuntime[AppContext]) -> str:
    """Lee preferencias persistidas del usuario"""
    item = runtime.store.get(("preferences",), runtime.context.user_id)
    return str(item.value) if item else "Sin preferencias guardadas."

@tool
def save_user_preference(key: str, value: str, runtime: ToolRuntime[AppContext]) -> str:
    """Guarda una preferencia persistente del usuario"""
    item = runtime.store.get(("preferences",), runtime.context.user_id)
    current = item.value if item else {}
    current[key] = value
    runtime.store.put(("preferences",), runtime.context.user_id, current)
    return f"Preferencia guardada: {key}={value}"

@tool
def find_runbook(topic: str, runtime: ToolRuntime[AppContext]) -> str:
    """Busca un runbook operativo simple."""
    rb = runtime.store.get(("runbooks",), "rb_502")
    if rb and ("502" in topic or "gateway" in topic.lower() or "runbook" in topic.lower()):
        data = rb.value
        return f"{data["title"]}\n- " + "\n- ".join(data["steps"])
    return f"No encontré runbook relevante para {topic}"

@tool
def read_internal_doc(doc_name: str, runtime: ToolRuntime[AppContext]) -> str:
    """Lee documentación interna simplificada."""
    doc = runtime.store.get(("docs",), "deploy_policy")
    if doc and any(word in doc_name.lower() for word in ["deploy", "policy", "política"]):
        data = doc.value
        return f"{data['title']}\n{data['content']}"
    return f"No encontré documento para: {doc_name}"

@tool
def architecture_notes(topic: str, runtime: ToolRuntime[AppContext]) -> str:
    """Devuelve notas de arquitecturas predefinidas."""
    notes = {
        "subagentes": "Úsalos cuando quieras control centralizado y especialistas como tools.",
        "handoffs": "Úsalo cuando el control conversacional cambie a otro agente.",
        "router": "Úsalo cuando la clasificación previa mejore costo, claridad y gobernanza."
    }
    topic_lower = topic.lower()
    for key, value in notes.items():
        if key in topic_lower:
            return value
    return "No encontré notas específicas; razona con principios generales."

# ============================================================
# 6) HELPERS
# ============================================================
def extract_last_user_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""

def extract_text(result: dict[str, Any]) -> str:
    final_message = result["messages"][-1]
    if isinstance(final_message, AIMessage):
        return str(final_message.content)
    return str(final_message)

def load_skill_text(store_obj: InMemoryStore, skill_name: str | None) -> str:
    if not skill_name:
        return ""
    item = store_obj.get(("skills",), skill_name)
    if not item:
        return ""
    skill = item.value
    few_shots = "\n".join(
        f"Ejemplo entrada: {fs["input"]}\nEjemplo salida: {fs["output"]}"
        for fs in skill.get("few_shots", [])
    )

    return (
        f"Skill cargada: {skill["name"]}\n"
        f"Instrucciones: {skill['instrucciones']}\n"
        f"{few_shots}"
    )

# ============================================================
# 7) MIDDLEWARE
# ============================================================
@dynamic_prompt
def build_prompt(request: ModelRequest) -> str:
    """
    Construye el prompt en función de:
    - ruta clasificada
    - skill cargada
    - preferencias del usuario
    - entorno
    """
    user_id = request.runtime.context.user_id
    prefs_item = request.runtime.store.get(("preferences",), user_id)
    prefs = prefs_item.value if prefs_item else {}

    state = request.runtime.state
    route: Route = state.get("route", "general")
    skill_name: str | None = state.get("skill_name")
    skill_text = load_skill_text(request.runtime.store, skill_name)

    parts = [
        "Eres un asistente técnico interno",
        f"Ruta activa: {route}",
        f"Entorno: {request.runtime.context.env}"
        "Debes responder con precisión y sin inventar datos."
    ]

    if prefs:
        parts.append(f"Preferencias del usuario: {prefs}")

    if route == "operations":
        parts.append("Prioriza diagnóstico breve, hipótesis y siguientes pasos.")
    elif route == "docs":
        parts.append("Prioriza estructura, resumen fiel y claridad documental.")
    elif route == "architecture":
        parts.append("Prioriza trade-offs, patrones y criterios de diseño.")

    if skill_text:
        parts.append(skill_text)

    return "\n\n".join(parts)

@wrap_model_call
def choose_model(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    route: Route = request.runtime.state.get("route", "general")
    model = deep_model if route in {"operations": "architecture"} else fast_model
    return handler(request.override(model=model))

@wrap_model_call
def choose_tools(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    route: Route = request.runtime.state.get("route", "general")

    allowed_by_route = {
        "operations": {"find_runbook", "get_user_preferences"},
        "docs": {"read_internal_doc", "get_user_preferences"},
        "architecture": {"architecture_notes", "get_user_preferences"},
        "general": {"get_user_preferences", "save_user_preference"},
    }

    allowed = allowed_by_route[route]
    filtered = [t for t in request.tools if getattr(t, "name", "") in allowed]
    return handler(request.override(tools=filtered))

# ============================================================
# 8) AGENTE ÚNICO
# ============================================================
agent = create_agent(
    model=fast_model,
    tools=[
        get_user_preferences,
        save_user_preference,
        find_runbook,
        read_internal_doc,
        architecture_notes,
    ],
    middleware=[build_prompt, choose_model, choose_tools],
    checkpointer=checkpointer,
    store=store,
    context_schema=AppContext,
)

# ============================================================
# 9) FUNCIÓN DE ENTRADA
# ============================================================
def ask_platform_assistant(
    user_input: str,
    *,
    thread_id: str,
    user_id: str,
    team_id: str = "platform",
    env: Literal["dev", "qa", "prod"] = "prod",
) -> str:
    route = route_request(user_input)
    skill_name = skill_for_route(route)
    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": user_input}],
            "route": route,
            "skill_name": skill_name,
        },
        config={"configurable": {"thread_id": thread_id}},
        context=AppContext(
            user_id=user_id,
            team_id=team_id,
            env=env,
        ),
    )
    return extract_text(result)

if __name__ == "__main__":
    q1 = "Tengo 502 intermitentes después de un despliegue. ¿Qué runbook sigo?"
    print(ask_platform_assistant(q1, thread_id="t-1", user_id="user-123"))

    q2 = "Resume la política de despliegues en 4 puntos."
    print(ask_platform_assistant(q2, thread_id="t-2", user_id="user-123"))

    q3 = "Para un sistema interno con API, ¿subagentes o handoffs?"
    print(ask_platform_assistant(q3, thread_id="t-3", user_id="user-123"))