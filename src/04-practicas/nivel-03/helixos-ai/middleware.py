# Prompt dinámico, modelo dinámico, tools dinámicas

from typing import Callable
from langchain.agents.middleware import wrap_model_call, dynamic_prompt, ModelResponse, ModelRequest
from langchain_openai import ChatOpenAI

from context import AppContext

@dynamic_prompt
def helix_dynamic_system_prompt(request: ModelRequest) -> str:
    """Construye el system prompt usando state, runtime context y store"""
    runtime = request.runtime
    state = request.state

    user_id = runtime.context.user_id
    tenant_id = runtime.context.tenant_id
    plan = runtime.context.plan

    style = "balanced"
    if runtime.store is not None:
        profile = runtime.store.get(("user_profile",), user_id)
        if profile:
            style = profile.value.get("communication_style", "balanced")
    route = state.get("route", "clarify")
    env = runtime.context.environment

    return f"""
Eres HelixOps AI, un agente senior de soporte Saas B2B.

Contexto del usuario:
- user_id: {user_id}
- tenant_id: {tenant_id}
- plan: {plan}
- environment: {env}
- route actual: {route}
- estilo de respuesta preferido: {style}

Reglas:
- Sé técnico, directo y accionable.
- Si el caso toca producción y hay acción destructiva, sugiere aprobación humana.
- Resume con pasos concretos y próximos riesgos.
""".strip()

@wrap_model_call
def select_model_and_tools(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Selecciona modelo y tools según complejidad, permisos y etapa."""
    state = request.state
    runtime = request.runtime

    message = state.get("user_message", "").lower()
    route = state.get("route", "clarify")
    severity = state.get("severity", "low")

    # Modelo dinámico
    # Ajusta estos nombres al provider que uses.
    if severity in {"high", "critical"} or route == "mixed":
        model = ChatOpenAI(model="gpt-4.1", temperature=0)
    else:
        model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    # Tool filtering dinámico
    allowed_tool_names = {
        "search_knowledge_base",
        "save_customer_preferences",
    }

    if route in {"billing", "mixed"}:
        allowed_tool_names.add("get_billing_status")

    if route in {"incident", "mixed"}:
        allowed_tool_names.update(
            {"search_incident_runbook", "inspect_service_health"}
        )
    if runtime.context.can_issue_refunds and "refund" in message:
        allowed_tool_names.add("issue_refund")

    tools = [
        tool_def for tool_def in request.tools if tool_def.name in allowed_tool_names
    ]

    return handler(request.override(model=model, tools=tools))