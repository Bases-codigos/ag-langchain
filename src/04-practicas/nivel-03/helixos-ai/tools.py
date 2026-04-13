# Vamos a modelar herramientas reales. Algunas son públicas, otras "sensibles".
# Luego las filtraremos dinámicamente.

from typing import Any
from langchain.tools import tool, ToolRuntime
from context import AppContext

@tool
def get_billing_status(invoice_id: str, runtime: ToolRuntime[AppContext]) -> str:
    """Obtiene el estado de una factura."""
    return f"Factura {invoice_id}: pagada, pero detectada doble imputación"

@tool
def issue_refund(invoice_id: str, reason: str, runtime: ToolRuntime[AppContext]) -> str:
    """Emite un reembolso. Solo para usuarios autorizados"""
    if not runtime.context.can_issue_refunds:
        return "No autorizado para emitir reembolsos."
    return f"Reembolso emitido para {invoice_id}. Motivo: {reason}"

@tool
def search_incident_runbook(service_name: str, runtime: ToolRuntime[AppContext]) -> str:
    """Busca el runbook operativo de un servicio"""
    return f"Runbook de {service_name}: verificar latencia, errores 5xx y webhooks fallidos."

@tool
def inspect_service_health(service_name: str, runtime: ToolRuntime[AppContext]) -> str:
    """Inspecciona el estado de salud de un servicio"""
    return f"Servicio {service_name}: 18% error rate, p95 alto, probable regresión reciente."

@tool
def search_knowledge_base(query: str, runtime: ToolRuntime[AppContext]) -> str:
    """Busca artículos internos de conocimientos"""
    return f"KB sobre '{query}': hay incidentes similares asociados a rotación de credenciales."

@tool
def save_customer_preferences(key: str, value: str, runtime: ToolRuntime[AppContext]) -> str:
    """Guarda preferencia del usuario en memoria larga."""
    assert runtime.store is not None
    runtime.store.put(
        ("user_preferences",),
        runtime.context.user_id,
        {key: value},
    )
    return f"Preferencia guardada: {key}={value}"

ALL_TOOLS = [
    get_billing_status,
    issue_refund,
    search_incident_runbook,
    inspect_service_health,
    search_knowledge_base,
    save_customer_preferences,
]