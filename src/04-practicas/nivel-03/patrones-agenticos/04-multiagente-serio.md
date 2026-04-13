# Multiagente Serio

## Que es

Es repartir el trabajo entre agentes con responsabilidades distintas, en vez de pedirle todo a un solo agente.

Un sistema multiagente serio no es solo "muchos prompts". Cada agente tiene un rol claro y produce salidas que otros agentes consumen.

## Agentes propuestos

### 1. Planner

Decide el plan de trabajo.

Preguntas que responde:

- que informacion hace falta
- que pasos conviene ejecutar
- que orden seguir

### 2. Extractor

Convierte texto libre en estructuras utiles.

Ejemplo:

- de mensaje del usuario a `EntradaIncidente`
- de incidente a `CasoExtraido`

### 3. Critic

Busca debilidades en la propuesta actual.

Ejemplo:

- falta evidencia
- la causa raiz es demasiado especulativa
- la resolucion es vaga

### 4. Verifier

Comprueba reglas o consistencia.

Ejemplo:

- el caso cumple politica de publicacion
- hay suficientes datos para guardar conocimiento
- la salida cumple formato esperado

### 5. Publisher

Se encarga de persistir, indexar y producir la salida final para el usuario o para la base de conocimiento.

## Como trabajan juntos

Secuencia tipica:

1. `Planner` define pasos.
2. `Extractor` estructura la informacion.
3. `Critic` intenta encontrar debilidades.
4. `Verifier` aplica reglas y validaciones.
5. `Publisher` guarda o responde.

## Ejemplo mental

Usuario:

"Desde anoche payments-api devuelve 500 y hay timeout de base de datos"

Pipeline:

- `Planner`: "extraer incidente, revisar completitud, buscar similares"
- `Extractor`: genera `EntradaIncidente` y `CasoExtraido`
- `Critic`: "falta evidencia sobre causa raiz"
- `Verifier`: "no cumple criterio de publicacion automatica"
- `Publisher`: responde pidiendo aclaracion o envia a revision

## Ejemplo simple con LangChain/LangGraph

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class MAState(TypedDict, total=False):
    texto_usuario: str
    plan: list[str]
    extracted_case: dict
    critic_notes: list[str]
    verified: bool
    respuesta: str


def planner(state: MAState) -> dict:
    return {"plan": ["extract", "critic", "verify", "publish"]}


def extractor(state: MAState) -> dict:
    return {
        "extracted_case": {
            "service": "payments-api",
            "symptom": "error 500",
        }
    }


def critic(state: MAState) -> dict:
    notes = []
    if "root_cause" not in state["extracted_case"]:
        notes.append("Falta causa raiz")
    return {"critic_notes": notes}


def verifier(state: MAState) -> dict:
    return {"verified": len(state.get("critic_notes", [])) == 0}


def publisher(state: MAState) -> dict:
    if state.get("verified"):
        return {"respuesta": "Caso verificado y listo para persistir"}
    return {"respuesta": "Se requiere mas evidencia antes de publicar"}


builder = StateGraph(MAState)
builder.add_node("planner", planner)
builder.add_node("extractor", extractor)
builder.add_node("critic", critic)
builder.add_node("verifier", verifier)
builder.add_node("publisher", publisher)
builder.add_edge(START, "planner")
builder.add_edge("planner", "extractor")
builder.add_edge("extractor", "critic")
builder.add_edge("critic", "verifier")
builder.add_edge("verifier", "publisher")
builder.add_edge("publisher", END)
graph = builder.compile()
```

## Cuando vale la pena

Este patron vale la pena cuando:

- una sola llamada al modelo ya no basta
- quieres separar responsabilidades
- necesitas revision tecnica interna
- necesitas mejorar confiabilidad

## Riesgo comun

Error frecuente: crear muchos agentes sin roles claros.

Si dos agentes hacen casi lo mismo, no ganaste arquitectura. Solo anadiste complejidad.

## Como llevarlo a tu proyecto

Una evolucion natural seria:

- `planner_agent`: decide si primero aclarar, recuperar o revisar
- `extractor_agent`: genera incidente y caso estructurado
- `critic_agent`: analiza lagunas y debilidades
- `verifier_agent`: valida politica y completitud
- `publisher_agent`: persiste e indexa si procede

## Regla practica

Usa multiagente cuando haya especializacion real.  
No cuando solo quieras repartir prompts sin una responsabilidad distinta por agente.
