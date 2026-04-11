from typing import Callable, Literal, Any
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, wrap_model_call, ModelRequest, ModelResponse
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain.messages import AIMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 1) CONTEXTO DE EJECUCIÓN
# ============================================================
@dataclass
class AppContext:
    user_id: str
    team_id: str
    environment: Literal["dev", "qa", "prod"]

# ============================================================
# 2) STORE Y CHECKPOINTER
# ============================================================
# Short-term memory: checkpointer por thread
# Long-term memory: store compartido entre conversaciones

checkpointer = InMemorySaver()
store = InMemoryStore()

# Seed inicial de memoria larga (preferencias del usuario)
store.put(
    ("preferences",),
    "user-123",
    {
        "communication_style": "concise",
        "prefer_bullets": True,
        "default_channel": "slack"
    },
)

store.put(
    ("runbooks", "payments",),
    "rb-502-gateway",
    {
        "service": "payments-api",
        "title": "HTTP 502 upstream troubleshooting",
        "steps": [
            "Verificar salud del upstream y balanceador",
            "Correlacionar timestamps con despliegues recientes",
            "Revisar timeouts y saturación del pool de conexiones",
            "Validar errores 5xx del servicio upstream",
        ],
        "severity_hint": "high"
    },
)

store.put(
    ("runbooks", "auth"),
    "rb-login-latency",
    {
        "service": "auth-service",
        "title": "Login latency troubleshooting",
        "steps": [
            "Revisar latencia del proveedor OIDC",
            "Validar consumo del servidor auth",
            "Inspeccionar consultas lentas a la base de datos",
        ],
        "severity_hint": "medium"
    }
)

# ============================================================
# 3) MODELOS DINÁMICOS
# ============================================================
# En producción aquí normalmente usarías un modelo rápido/barato
# para tareas simples y uno más fuerte para análisis complejos.
simple_model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
complex_model = ChatOpenAI(model="gpt-4.1", temperature=0)

# ============================================================
# 4) TOOLS BASE
# ============================================================
@tool
def search_recent_incidents(service: str) -> str:
    """Busca incidentes recientes de servicio y devuelve un resumen textual"""
    fake_db = {
        "payments-api": [
            "INC-201: 502 intermitente después de despliegue canary",
            "INC-187: picos de timeout por saturación del pool HTTP",
        ],
        "auth-service": [
            "INC-155: aumento de latencia por dependencia OIDC"
        ],
    }
    incidents = fake_db.get(service, [])
    if not incidents:
        return f"No encontré incidentes recientes para {service}."
    return "\n".join(f"- {item}" for item in incidents)

@tool
def get_service_health(service: str) -> str:
    """Devuelve una foto rápida de salud del servicio."""
    fake_health = {
        "payment-api": "status=degraded, error_rate=3.1%, p95=1.8s, saturation=high",
        "auth-service": "status=warning, error_rate=0.4%, p95=950ms, saturation=medium",
    }
    return fake_health.get(service, f"No hay métricas disponibles para {service}.")

@tool
def search_runbooks(service: str, symptom: str) -> str:
    """Busca runbooks en memoria larga según el servicio y síntoma"""
    namespace_candidates = [
        ("runbooks", "payments"),
        ("runbooks", "auth"),
    ]
    matches: list[str] = []
    symptoms_lower = symptom.lower()
    service_lower = service.lower()

    for namespace in namespace_candidates:
        # En un store real, aquí usarías búsqueda más robusta
        for key in ["rb-502-gateway", "rb-login-latency"]:
            item = store.get(namespace, key)
            if item is None:
                continue

            value = item.value
            haystack = f"{value["service"]} {value['title']} {" ".join(value["steps"])}".lower()
            if service_lower in haystack or symptoms_lower in haystack:
                matches.append(
                    f"Runbook: {value['title']}\n"
                    f"Service: {value['service']}\n"
                    f"Severity hint: {value['severity_hint']}\n"
                    f"Steps:\n- " + "\n- ".join(value["steps"])
                )
    if not matches:
        return f"No encontré runbooks para service={service}, symptom={symptom}."

    return "\n\n".join(matches)

@tool
def save_user_preference(key: str, value: str, runtime: ToolRuntime[AppContext]) -> str:
    """Guarda preferencia del usuario en memoria larga"""
    user_id = runtime.context.user_id

    current = store.get(("preferences", ), user_id)
    current_value = current.value if current else {}

    current_value[key] = value
    store.put(("preferences", ), user_id, current_value)

    return f"Preferencia guardada: {key}={value}"

@tool
def get_user_preferences(runtime: ToolRuntime[AppContext]) -> str:
    """Lee las preferencias persistentes del usuario"""
    user_id = runtime.context.user_id
    prefs = store.get(("preferences", ), user_id)

    if prefs is None:
        return "El usuario no tiene preferencias guardadas"

    return str(prefs.value)

@tool
def create_status_update(channel: str, summary: str) -> str:
    """Simula la creación de un mensaje para Slack/Email/Jira."""
    return f"Draft creado para canal={channel}: \n{summary}"

# ============================================================
# 5) HELPERS
# ============================================================
def _last_user_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""

def extract_text(result: dict[str, Any]) -> str:
    final_message = result["messages"][-1]
    if isinstance(final_message, AIMessage):
        return str(final_message.content)
    return str(final_message)

def _looks_complex(text: str) -> bool:
    keywords = [
        "analiza",
        "causa raíz",
        "root cause",
        "incidente",
        "error",
        "latencia",
        "arquitectura",
        "compara",
        "runbook",
    ]
    lower = text.lower()
    return any(keyword in lower for keyword in keywords) or len(text) > 180

# ============================================================
# 6) MIDDLEWARE DEL SUPERVISOR
# ============================================================
@dynamic_prompt
def supervisor_prompt(request: ModelRequest) -> str:
    """
    Prompt dinámico basado en:
    - estado corto (cantidad de mensajes)
    - store largo (preferencias del usuario)
    - contexto estático (entorno)
    """
    base = [
        "Eres un supervisor multiagente para incidentes de plataforma.",
        "Tu trabajo es delegar en especialistas cuando sea necesario.",
        "No inventes métricas ni causas; si algo no se sabe, dilo con claridad.",
        "Prioriza pasos accionables y próximos pasos concretos.",
    ]
    # Contexto estático
    env_name = request.runtime.context.environment
    base.append(f"Entorno actual: {env_name}")

    # Estado corto
    if len(request.messages) > 8:
        base.append("La conversación ya es larga: responde de forma más compacta.")

    # Store largo
    user_id = request.runtime.context.user_id
    prefs = store.get(("preferences", ), user_id)
    if prefs:
        style = prefs.value.get("communication_style", "balanced")
        prefs_bullets = prefs.value.get("prefer_bullets", False)
        base.append(f"El usuario prefiere respuestas de estilo: {style}")
        if prefs_bullets:
            base.append("Usa viñetas cuando ayuden a claridad.")

    return "\n".join(base)

@wrap_model_call
def choose_model(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelRequest],
) -> ModelRequest:
    """
    Cambia de modelo según complejidad
    """
    latest_tex = _last_user_text(request.messages)
    model = complex_model if _looks_complex(latest_tex) else simple_model
    return handler(request.override(model=model))

@wrap_model_call
def choose_tools(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelRequest],
) -> ModelRequest:
    """
    Filtra herramientas expuestas al supervisor.
    Esto reduce ruido, costo y errores de tool selection
    """
    latest_text =_last_user_text(request.messages).lower()
    all_tools = list(request.tools)

    def tool_name(tool_obj: Any) -> str:
        return getattr(tool_obj, "name", tool_obj)

    if any(word in latest_text for word in ["preferencia", "recuerda", "guardar estilo"]):
        allowed = {"save_user_preference", "get_user_preferences"}
    elif any(word in latest_text for word in ["resume", "redacta", "slack", "correo", "status update"]):
        allowed = {"incident_comms"}
    elif any(word in latest_text for word in ["runbook", "pasos", "troubleshooting"]):
        allowed = {"runbook_specialist"}
    elif any(word in latest_text for word in ["analiza", "causa", "error", "incidente", "latencia", "502"]):
        allowed = {"incident_triage"}
    else:
        # fallback: dejar supervisor + utilidades básicas
        allowed = {
            "incident_triage",
            "runbook_specialist",
            "incident_comms",
            "get_user_preferences",
        }
    filtered = [tool_obj for tool_obj in all_tools if tool_name(tool_obj) in allowed]
    return handler(request.override(tools=filtered))

# ============================================================
# 7) SUBAGENTES ESPECIALISTAS
# ============================================================
triage_agent = create_agent(
    model=complex_model,
    tools=[search_recent_incidents, get_service_health],
    system_prompt=(
        "Eres un asistente especialista de triage de incidentes.\n"
        "Analiza síntomas, severidad probable, hipótesis y señales observables.\n"
        "No redactes mensajes ejecutivos largos.\n"
        "Tu salida final debe incluir:\n"
        "1. evaluación breve\n"
        "2. hipótesis\n"
        "3. siguiente paso recomendado"
    ),
)

runbook_agent = create_agent(
    model=simple_model,
    tools=[search_runbooks],
    system_prompt=(
        "Eres un especialista en runbooks operativos.\n"
        "Debes elegir el procedimiento más útil ára el síntoma descrito.\n"
        "Tu salida final debe devolver pasos concreos, en orden, sin relleno."
    ),
)

comms_agent = create_agent(
    model=simple_model,
    tools=[create_status_update, get_service_health],
    system_prompt=(
        "Eres un especialista en comunicación de incidentes.\n"
        "Transforma hallazgos técnicos en mensajes claros para Slack, correo o ticket.\n"
        "Sé profesional, preciso y accionable."
    ),
)

# ============================================================
# 8) SUBAGENTES ENVUELTOS COMO TOOLS DEL SUPERVISOR
# ============================================================
@tool("incident_triage")
def incident_triage(request: str, runtime: ToolRuntime[AppContext]) -> str:
    """
    Analiza un incidente técnico
    """
    # Pasamos contexto mínimo sintetizado, no el historial completo.
    original_user_text = _last_user_text(runtime.state["messages"])
    prompt = (
        "Consulta original del usuario:\n"
        f"{original_user_text}\n\n"
        "Subtarea asignada al especialista de triage:\n"
        f"{request}"
    )

    result = triage_agent.invoke({
        "messages": [{"role": "user", "content": prompt}],
    })
    return extract_text(result)

@tool("runbook_specialist")
def runbook_specialist(request: str, runtime: ToolRuntime[AppContext]) -> str:
    """
    Busca y propone runbooks
    """
    original_user_text = _last_user_text(runtime.state["messages"])
    prompt = (
        "Consulta original del usuario:\n"
        f"{original_user_text}\n\n"
        "Subtarea asignada al especialista de runbooks:\n"
        f"{request}"
    )

    result = runbook_agent.invoke({
        "messages": [{"role": "user", "content": prompt}],
    })

    return extract_text(result)

@tool("incident_comms")
def incident_comms(request: str, runtime: ToolRuntime[AppContext]) -> str:
    """
    Redacta actualizaciones o resúmenes del incidente
    """
    original_user_text = _last_user_text(runtime.state["messages"])
    prompt = (
        "Consulta original del usuario:\n"
        f"{original_user_text}\n\n"
        "Subtarea asignada al especialista de comunicación:\n"
        f"{request}"
    )
    result = comms_agent.invoke({
        "messages": [{"role": "user", "content": prompt}],
    })
    return extract_text(result)

# ============================================================
# 9) SUPERVISOR
# ============================================================
supervisor_agent = create_agent(
    model=simple_model,
    tools=[
        incident_triage,
        runbook_specialist,
        incident_comms,
        save_user_preference,
        get_user_preferences,
    ],
    middleware=[
        supervisor_prompt,
        choose_model,
        choose_tools,
    ],
    checkpointer=checkpointer,
    store=store,
    system_prompt=(
        "Eres el supervisor principal.\n"
        "Puedes delegar a especialistas y combinar resultados.\n"
        "Si el usuario pide análisis + pasos + redacción, puees encadenar varios especialistas.\n"
        "Mantén el control centralizado en la conversación"
    ),
)

# ============================================================
# 10) FUNCIÓN DE USO
# ============================================================
def ask_incident_assistant(
    user_input: str,
    *,
    thread_id: str,
    user_id: str,
    team_id: str = "platform",
    environment: Literal["dev", "qa", "prod"] = "prod",
) -> str:
    result = supervisor_agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config={"configurable": {"thread_id": thread_id}},
        context=AppContext(
            user_id=user_id,
            team_id=team_id,
            environment=environment,
        ),
    )
    return extract_text(result)

if __name__ == "__main__":
    # Turno 1
    response_1 = ask_incident_assistant(
        (
            "Analiza este incidente: payments-api tiene 502 intermitentes "
            "después de un despliegue. Dame hipótesis y el runbook más probable."
        ),
        thread_id="inc-9001",
        user_id="user-123",
    )
    print("=== RESPUESTA 1 ===")
    print(response_1)

    # Turno 2: misma conversación, usa short-term memory del thread
    response_2 = ask_incident_assistant(
        "Ahora redáctame un status update corto para Slack.",
        thread_id="inc-9001",
        user_id="user-123",
    )
    print("\n=== RESPUESTA 2 ===")
    print(response_2)

    # Turno 3: persistencia en memoria larga del usuario
    response_3 = ask_incident_assistant(
        "Guarda que prefiero respuestas ultra breves.",
        thread_id="prefs-1",
        user_id="user-123",
    )
    print("\n=== RESPUESTA 3 ===")
    print(response_3)
