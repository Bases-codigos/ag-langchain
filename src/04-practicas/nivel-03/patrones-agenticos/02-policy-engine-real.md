# Policy Engine Real

## Que es

Es una capa separada que decide si una accion esta permitida o no.

No debe vivir solo dentro del prompt. Debe ser codigo o reglas ejecutables.

## Para que sirve

- centralizar permisos y restricciones
- evitar que el modelo decida cosas sensibles por su cuenta
- aplicar reglas por tenant, entorno, riesgo o rol
- justificar por que una accion fue bloqueada o aprobada

## Ejemplos de politicas

- en `prod`, publicar conocimiento requiere aprobacion humana
- si el `risk_score` > 0.8, no se permite auto-publicacion
- un usuario `viewer` no puede aprobar casos
- si falta evidencia, no se puede pasar a `publicado`

## Ejemplo simple

```python
from dataclasses import dataclass


@dataclass
class PolicyContext:
    role: str
    environment: str
    risk_score: float


def can_publish(ctx: PolicyContext, has_evidence: bool) -> tuple[bool, str]:
    if not has_evidence:
        return False, "Falta evidencia suficiente"

    if ctx.environment == "prod" and ctx.role != "reviewer":
        return False, "En prod solo reviewer puede publicar"

    if ctx.risk_score > 0.8:
        return False, "Riesgo demasiado alto para auto-publicacion"

    return True, "Permitido"
```

## Integracion con LangGraph

```python
from typing_extensions import TypedDict


class ReviewState(TypedDict, total=False):
    has_evidence: bool
    approved: bool
    policy_reason: str


def aplicar_policy(state: ReviewState, runtime) -> dict:
    allowed, reason = can_publish(
        ctx=runtime.context,
        has_evidence=state.get("has_evidence", False),
    )
    return {
        "approved": allowed,
        "policy_reason": reason,
    }
```

## Diferencia entre policy engine y prompt

Prompt:

- "si estas en produccion, intenta ser cuidadoso"

Policy engine:

- "si environment == prod y role != reviewer, bloquear publicacion"

El prompt orienta. La policy decide.

## Como se combina con agentes

- el agente propone una accion
- el policy engine valida si esta permitida
- el flujo decide continuar, pausar o rechazar

## Como llevarlo a tu proyecto

Podrias crear un archivo `policy.py` con reglas como:

- `can_request_more_info(...)`
- `can_publish_case(...)`
- `requires_human_review(...)`
- `can_edit_case(...)`

Y en el grafo hacer que:

- primero el agente propone
- luego policy decide
- despues el grafo continua

## Regla practica

Las decisiones sensibles no deben depender solo del LLM. Deben pasar por una capa de politica separada.
