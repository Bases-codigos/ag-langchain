# Motor De Estado Formal

## Que es

Es una forma de modelar un proceso con:

- estados validos
- transiciones validas
- reglas de negocio que impiden saltos incorrectos

No es solo "si pasa A, ve a B". Es decir explicitamente en que fase esta el caso y que cambios estan permitidos.

## Para que sirve

- evitar flujos inconsistentes
- hacer el sistema mas predecible
- poder auditar por que cambio un caso de una fase a otra
- soportar crecimiento sin volver el flujo caotico

## Idea practica

En vez de tener solo un `state` con datos, anades un campo como `case_status`:

```python
case_status: Literal[
    "nuevo",
    "clasificado",
    "en_aclaracion",
    "listo_para_revision",
    "aprobado",
    "rechazado",
    "publicado",
]
```

Y luego validas que no ocurra algo invalido, por ejemplo:

- no puedes publicar si no paso por revision
- no puedes aprobar si el caso no esta listo para revision

## Ejemplo simple con LangGraph

```python
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


class CaseState(TypedDict, total=False):
    case_status: Literal[
        "nuevo",
        "clasificado",
        "listo_para_revision",
        "publicado",
    ]
    severity: Literal["low", "high"]
    approved: bool


def clasificar(state: CaseState) -> Command[Literal["revision", "publicar"]]:
    if state["case_status"] != "nuevo":
        raise ValueError("Transicion invalida: solo se clasifica desde 'nuevo'")

    if state["severity"] == "high":
        return Command(
            update={"case_status": "listo_para_revision"},
            goto="revision",
        )

    return Command(
        update={"case_status": "clasificado"},
        goto="publicar",
    )


def revision(state: CaseState) -> dict:
    if state["case_status"] != "listo_para_revision":
        raise ValueError("Transicion invalida: revision fuera de fase")
    return {"approved": True}


def publicar(state: CaseState) -> dict:
    if state.get("approved") is False:
        raise ValueError("No se puede publicar sin aprobacion")
    return {"case_status": "publicado"}


builder = StateGraph(CaseState)
builder.add_node("clasificar", clasificar)
builder.add_node("revision", revision)
builder.add_node("publicar", publicar)
builder.add_edge(START, "clasificar")
builder.add_edge("revision", "publicar")
builder.add_edge("publicar", END)
graph = builder.compile()
```

## Que mejora este patron

Sin este patron:

- el sistema cambia de etapa sin control fuerte
- el bug aparece tarde
- los estados invalidos se mezclan

Con este patron:

- el error aparece donde se produjo
- las reglas del dominio quedan explicitas
- es mas facil mantener y extender el flujo

## Como llevarlo a tu proyecto

En un proyecto de incidentes, podrias anadir:

- `case_status`
- `last_transition_reason`
- `transition_history`

Y validar transiciones como:

- `nuevo -> en_aclaracion`
- `en_aclaracion -> listo_para_revision`
- `listo_para_revision -> aprobado | rechazado`
- `aprobado -> publicado`

## Regla practica

Si un proceso tiene fases del negocio, no las escondas dentro de prompts o nombres de nodos. Modelalas como estado formal.
