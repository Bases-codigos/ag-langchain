from dataclasses import dataclass
from typing_extensions import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime


# =========================================================
# 1) Estado del grafo
# =========================================================
class State(TypedDict):
    input: str
    results: str


# =========================================================
# 2) Contexto de ejecución
#    Este contexto NO forma parte del estado.
#    Se pasa por separado en graph.invoke(..., context=...)
# =========================================================
@dataclass
class Context:
    user_id: str


# =========================================================
# 3) Crear el builder
#    context_schema permite tipar runtime.context
# =========================================================
builder = StateGraph(State, context_schema=Context)


# =========================================================
# 4) Nodo simple: solo recibe el estado
# =========================================================
def plain_node(state: State):
    print("En plain_node, input =", state["input"])
    return {
        "results": f"plain_node recibió: {state['input']}"
    }


# =========================================================
# 5) Nodo con runtime: accede al contexto
# =========================================================
def node_with_runtime(state: State, runtime: Runtime[Context]):
    print("En node_with_runtime, user_id =", runtime.context.user_id)
    return {
        "results": f"Hola, {state['input']}! user_id={runtime.context.user_id}"
    }


# =========================================================
# 6) Nodo con config: accede al RunnableConfig
# =========================================================
def node_with_config(state: State, config: RunnableConfig):
    thread_id = config.get("configurable", {}).get("thread_id", "sin-thread-id")
    print("En node_with_config, thread_id =", thread_id)
    return {
        "results": f"Hola, {state['input']}! thread_id={thread_id}"
    }


# =========================================================
# 7) Registrar nodos
# =========================================================
builder.add_node("plain_node", plain_node)
builder.add_node("node_with_runtime", node_with_runtime)
builder.add_node("node_with_config", node_with_config)


# =========================================================
# 8) Conectar el grafo
# =========================================================
builder.add_edge(START, "plain_node")
builder.add_edge("plain_node", "node_with_runtime")
builder.add_edge("node_with_runtime", "node_with_config")
builder.add_edge("node_with_config", END)


# =========================================================
# 9) Compilar
# =========================================================
graph = builder.compile()


# =========================================================
# 10) Ejecutar
# =========================================================
result = graph.invoke(
    {"input": "Harold"},
    config={"configurable": {"thread_id": "thread-123"}},
    context=Context(user_id="user-999"),
)

print("\nResultado final:")
print(result)