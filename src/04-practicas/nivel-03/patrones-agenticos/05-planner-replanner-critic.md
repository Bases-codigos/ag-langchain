# Planner + Replanner + Critic

Este documento explica dos versiones del mismo patron:

1. Version didactica: minima y facil de entender.
2. Version real: con `structured output`, control de riesgos y condiciones de parada robustas.

## 1) Version didactica

## Objetivo

Modelar un ciclo de trabajo donde el sistema:

- crea un plan inicial
- ejecuta un paso
- evalua resultado
- ajusta el plan si hace falta

## Roles

- `Planner`: propone pasos ordenados.
- `Executor`: ejecuta el paso actual.
- `Critic`: valida si el resultado del paso es util o insuficiente.
- `Replanner`: modifica el plan cuando el critic detecta problemas.

## Estado minimo recomendado

```python
class AgentState(TypedDict, total=False):
    objective: str
    plan: list[dict]           # pasos
    current_index: int         # puntero al paso actual
    last_result: str           # ultimo resultado de ejecucion
    critic_verdict: str        # ok | replan | fail
    goal_status: str           # in_progress | done | failed
    replan_count: int
    max_replans: int
```

## Ejemplo simple en LangGraph

```python
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

class Step(TypedDict):
    id: str
    action: str
    status: Literal["pending", "done", "failed"]

class AgentState(TypedDict, total=False):
    objective: str
    plan: list[Step]
    current_index: int
    last_result: str
    critic_verdict: Literal["ok", "replan", "fail"]
    goal_status: Literal["in_progress", "done", "failed"]
    replan_count: int
    max_replans: int

def planner(state: AgentState) -> dict:
    return {
        "plan": [
            {"id": "s1", "action": "buscar_datos", "status": "pending"},
            {"id": "s2", "action": "analizar_datos", "status": "pending"},
            {"id": "s3", "action": "ejecutar_accion", "status": "pending"},
        ],
        "current_index": 0,
        "goal_status": "in_progress",
        "replan_count": 0,
        "max_replans": 3,
    }

def executor(state: AgentState) -> dict:
    step = state["plan"][state["current_index"]]
    if step["action"] == "ejecutar_accion" and state.get("replan_count", 0) == 0:
        result = "fallo: falta_validacion"
        step["status"] = "failed"
    else:
        result = "ok"
        step["status"] = "done"
    plan = state["plan"]
    plan[state["current_index"]] = step
    return {"plan": plan, "last_result": result}

def critic(state: AgentState) -> dict:
    if state["last_result"].startswith("fallo:"):
        return {"critic_verdict": "replan"}
    all_done = all(s["status"] == "done" for s in state["plan"])
    if all_done:
        return {"critic_verdict": "ok", "goal_status": "done"}
    return {"critic_verdict": "ok", "goal_status": "in_progress"}

def replanner(state: AgentState) -> dict:
    count = state["replan_count"] + 1
    if count > state["max_replans"]:
        return {"goal_status": "failed", "replan_count": count}

    plan = state["plan"]
    needs_validation = not any(s["action"] == "validar_permiso" for s in plan)
    if needs_validation:
        plan.insert(2, {"id": "s2b", "action": "validar_permiso", "status": "pending"})

    next_idx = next((i for i, s in enumerate(plan) if s["status"] == "pending"), len(plan) - 1)
    return {"plan": plan, "current_index": next_idx, "replan_count": count, "goal_status": "in_progress"}

def router_after_critic(state: AgentState) -> Literal["advance", "replanner", "end"]:
    if state.get("goal_status") == "done":
        return "end"
    if state["critic_verdict"] == "replan":
        return "replanner"
    return "advance"

def advance(state: AgentState) -> Command[Literal["executor", "end"]]:
    next_idx = state["current_index"] + 1
    if next_idx >= len(state["plan"]):
        return Command(update={"goal_status": "done"}, goto="end")
    return Command(update={"current_index": next_idx}, goto="executor")

builder = StateGraph(AgentState)
builder.add_node("planner", planner)
builder.add_node("executor", executor)
builder.add_node("critic", critic)
builder.add_node("replanner", replanner)
builder.add_node("advance", advance)
builder.add_node("end", lambda s: s)
builder.add_edge(START, "planner")
builder.add_edge("planner", "executor")
builder.add_edge("executor", "critic")
builder.add_conditional_edges(
    "critic",
    router_after_critic,
    {"advance": "advance", "replanner": "replanner", "end": "end"},
)
builder.add_edge("replanner", "executor")
builder.add_edge("end", END)
graph = builder.compile()
```

## 2) Version real (produccion)

La version real mantiene la misma arquitectura, pero cambia 4 cosas clave:

- usa salidas estructuradas del LLM para `planner`, `critic`, `replanner`
- valida que el plan sea ejecutable
- introduce politicas y limites de seguridad
- registra trazas y evidencia por paso

## Esquemas estructurados

```python
from typing import Literal
from pydantic import BaseModel, Field

class PlanStep(BaseModel):
    id: str
    action: str
    tool_name: str | None = None
    input: dict = Field(default_factory=dict)
    success_criteria: str
    depends_on: list[str] = Field(default_factory=list)

class PlanOut(BaseModel):
    steps: list[PlanStep]
    rationale: str

class CriticOut(BaseModel):
    verdict: Literal["ok", "replan", "fail"]
    reason: str
    missing_evidence: list[str] = Field(default_factory=list)
    risk_score: float = Field(ge=0.0, le=1.0)

class ReplanOut(BaseModel):
    action: Literal["insert_steps", "replace_remaining", "abort"]
    steps: list[PlanStep] = Field(default_factory=list)
    reason: str
```

## Flujo real recomendado

1. `planner_llm` crea `PlanOut`.
2. `validate_plan` comprueba:
- no hay pasos duplicados
- no hay dependencias circulares
- las tools existen
- no hay pasos prohibidos por politica
3. `executor` corre solo un paso y produce evidencia.
4. `critic_llm` devuelve `CriticOut`.
5. `policy_gate` aplica reglas duras:
- si `risk_score` alto en `prod` -> human approval
- si falta evidencia critica -> no publica
6. `replanner_llm` ajusta plan con `ReplanOut`.
7. loop hasta `done`, `failed` o `abort`.

## Ejemplo corto de planner real

```python
def planner_llm(state, runtime):
    planner = modelo_llm.with_structured_output(PlanOut)
    prompt = f"""
Objetivo: {state["objective"]}
Devuelve un plan minimo ejecutable.
Incluye criterios de exito por paso.
No uses tools no permitidas.
"""
    out = planner.invoke(prompt)
    return {"plan": [s.model_dump() for s in out.steps], "plan_rationale": out.rationale}
```

## Ejemplo corto de critic real

```python
def critic_llm(state, runtime):
    critic = modelo_llm.with_structured_output(CriticOut)
    prompt = f"""
Paso ejecutado: {state["current_step"]}
Resultado: {state["last_result"]}
Evidencia: {state.get("evidence", [])}
Evalua si el objetivo sigue avanzando.
"""
    out = critic.invoke(prompt)
    return {
        "critic_verdict": out.verdict,
        "critic_reason": out.reason,
        "missing_evidence": out.missing_evidence,
        "risk_score": out.risk_score,
    }
```

## Ejemplo corto de replanner real

```python
def replanner_llm(state, runtime):
    replanner = modelo_llm.with_structured_output(ReplanOut)
    prompt = f"""
Plan actual: {state["plan"]}
Paso fallido: {state["current_step"]}
Critica: {state["critic_reason"]}
Faltantes: {state.get("missing_evidence", [])}
Propone ajuste minimo para avanzar.
"""
    out = replanner.invoke(prompt)

    if out.action == "abort":
        return {"goal_status": "failed", "abort_reason": out.reason}

    return {
        "replan_action": out.action,
        "replan_steps": [s.model_dump() for s in out.steps],
        "replan_reason": out.reason,
    }
```

## Guardrails que no deberian faltar

- `max_steps` y `max_replans` para evitar loops.
- `allowed_tools` por entorno y rol.
- verificacion de `success_criteria` por paso.
- bloqueo de acciones destructivas sin aprobacion.
- fallback a humano si el critic marca riesgo alto repetido.

## Como se gestiona el orden de ejecucion en real

- orden inicial: lo propone `planner`
- orden efectivo: lo impone el grafo con `current_index`
- orden actualizado: lo modifica `replanner`
- orden final: queda registrado en historial con timestamps y resultados

No se "improvisa" el orden en cada prompt; se escribe y controla en el estado.

## Criterio de calidad del patron

Un buen sistema planner/replanner/critic cumple:

- produce planes cortos y ejecutables
- explica por que replanifica
- evita loops infinitos
- registra evidencia por paso
- mantiene trazabilidad entre plan, ejecucion y salida final

## Siguiente paso recomendado

Implementar este patron sobre tu flujo de incidentes:

- `planner` decide ruta (aclarar, recuperar, revisar)
- `critic` evalua si hay evidencia para publicar
- `replanner` inserta pasos de aclaracion o validacion antes de persistir
