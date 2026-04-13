from typing import Callable, Any, Literal
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, dynamic_prompt, wrap_model_call
from langchain.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

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
checkpointer = InMemoryStore()
store = InMemoryStore()

# Seed inicial de memoria larga (preferencia del usuario)
store.put(
    ("preferences",),
    "user_123",
    {
        "communication_style": "concise",
        "prefer_bullets": True,
        "default_channel": "slack",
    },
)

store.put(
    ("runbooks", "payments",),
    "rb-502-gateway",
    {
        "service": "payment-api",
        "title": "HTTP 502 upstream troubleshooting",
        "steps": [
            "Verificar salud del upstream y balanceador",
            "Correlacionar timestamps con despliegues recientes",
            "Revisar timeouts y saturación del pool de conexiones",
            "Validar errores 5xx del servicio upstream"
        ],
        "severity_hint": "high"
    },
)

store.put(
    ("runbooks", "auth",),
    "rb-login-latency",
    {
        "service": "auth-service",
        "title": "Login latency troubleshooting",
        "steps": [
            "Revisar latencia del proveedor OIDC",
            "Validar consumo de CPU del servicio auth",
            "Inspeccionar consultas lentas a la base de datos",
        ],
        "severity_hint": "medium"
    }
)

# ============================================================
# 3) MODELOS DINÁMICOS
# ============================================================
# En producción aquí normalmente se usa un modelo rápido/barato
# para tareas simples y uno más fuerte para análisis complejos.

simple_model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
complex_model = ChatOpenAI(model="gpt-4.1", temperature=1)

# ============================================================
# 4) TOOLS BASE
# ============================================================
@tool
def search_recent_incidents(service: str) -> str:
    """
    Busca incidentes recientes de un servicio y devuelve un resumen textual.
    """
    fake_db = {
        "payments-api": [
            "INC-201: 502 intermitente después de despliegues canary",
            "INC-187: picos de timeout por saturación del pool HTTP",
        ],
        "auth-service": [
            "INC-155: aumento de latencia por dependencia OIDC",
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
        "payments-api": "status=degraded, error_rate=3.1%, p95=1.8s, saturation=high",
        "auth-service": "status=warning, error_rate=0.4%, p95=950ms, saturation=medium",
    }
    return fake_health.get(service, f"No hay métricas disponibles para {service}.")

@tool
def search_runbooks(service: str, symptom: str) -> str:
    """Busca runbooks en memoria larga según servicio y síntoma."""
    namespace_candidates = [
        ("runbooks", "payments"),
        ("runbooks", "auth"),
    ]

    matches: list[str] = []
    symptom_lower = symptom.lower()
    service_lower = service.lower()

    for namespace in namespace_candidates:
        # En un store real, aquí usarías búsqueda más robusta.
        for key in ["rb-502-gateway", "rb-login-latency"]:
            item = store.get(namespace, key)
            if item is None:
                continue

            value = item.value
            haystack = f"{value["service"]} {value['title']} {' '.join(value["steps"])}".lower()
            if service_lower in haystack or symptom_lower in haystack:
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
    """Guarda una preferencia del usuario en memoria larga."""
    user_id = runtime.context.user_id

    current = store.get(("preferences",), user_id)
    current_value = current.value if current else {}

    current_value[key] = value
    store.put(("preferences",), user_id, current_value)

    return f"Preferencia guardada: {key}={value}"

@tool
def get_user_preferences(runtime: ToolRuntime[AppContext]) -> str:
    """Lee las preferencias persistidas del usuario."""
    user_id = runtime.context.user_id
    prefs = store.get(("preferences",), user_id)

    if prefs is None:
        return "El usuario no tiene preferencias guardadas."

    return str(prefs.value)

@tool
def create_status_update(channel: str, summary: str) -> str:
    """Simula la creación de un mensaje para Slack/Email/Jira."""
    return f"Draft creado para canal={channel}:\n{summary}"

# ============================================================
# 5) HELPERS
# ============================================================












