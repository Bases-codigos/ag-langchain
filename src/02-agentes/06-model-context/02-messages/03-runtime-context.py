from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelResponse, ModelRequest
from typing import Callable
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Context:
    user_jurisdiction: str
    industry: str
    compliance_frameworks: list[str]

@wrap_model_call
def inject_compliance_rules(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Inyectar restricciones de cumplimiento desde el contexto de ejecución."""
    jurisdiction = request.runtime.context.user_jurisdiction
    industry = request.runtime.context.industry
    frameworks = request.runtime.context.compliance_frameworks

    rules = []

    if "GDPR" in frameworks:
        rules.append("- Debe obtener consentimiento explícito antes de tratar datos personales")
        rules.append("- Los usuarios tienen derecho a la eliminación de datos")
    if "HIPAA" in frameworks:
        rules.append("- No se puede compartir información sanitaria del paciente sin autorización")
        rules.append("- Debe usar comunicación segura y cifrada")
    if industry == "finance":
        rules.append("- No se puede ofrecer asesoramiento financiero sin las exenciones de responsabilidad adecuadas")

    if rules:
        compliance_context = f"""Requisitos de cumplimiento para {jurisdiction}:
{chr(10).join(rules)}"""

        # Append at end - models pay more attention to final messages
        messages = [
            *request.messages,
            {"role": "user", "content": compliance_context}
        ]
        request = request.override(messages=messages)

    return handler(request)

agent = create_agent(
    model="gpt-4.1",
    tools=[...],
    middleware=[inject_compliance_rules],
    context_schema=Context
)