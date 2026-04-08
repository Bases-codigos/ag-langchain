from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

# =========================================================
# 1) Esquema de entrada del grafo
#    Solo contiene lo que el usuario puede enviar al invocar el grafo
# =========================================================
class InputState(TypedDict):
    user_input: str

# =========================================================
# 2) Esquema de salida del grafo
#    Solo contiene lo que queremos devolver al final
# =========================================================
class OutputState(TypedDict):
    graph_output: str

# =========================================================
# 3) Esquema interno completo del grafo
#    Contiene todos los canales que usan los nodos internamente
# =========================================================
class OverallState(TypedDict):
    foo: str
    user_input: str
    graph_output: str

# =========================================================
# 4) Esquema privado
#    Se usa para comunicación interna entre nodos
#    y no forma parte de la entrada o salida pública del grafo
# =========================================================
class PrivateState(TypedDict):
    bar: str

# =========================================================
# 5) Nodo 1
#    Lee desde el esquema de entrada y escribe al estado interno
# =========================================================
def node_1(state: InputState) -> OverallState:
    # Escribe en el estado interno del grafo
    return {
        "foo": state["user_input"] + " name"
    }

# =========================================================
# 6) Nodo 2
#    Lee del estado interno y escribe al estado privado
# =========================================================
def node_2(state: OverallState) -> PrivateState:
    # Lee del estado interno y escribe en el estado privado
    return {
        "bar": state["foo"] + " is"
    }

# =========================================================
# 7) Nodo 3
#    Lee del estado privado y escribe al esquema de salida
# =========================================================
def node_3(state: PrivateState) -> OutputState:
    # Lee del estado privado y escribe en la salida final
    return {
        "graph_output": state["bar"] + " Lance"
    }

# =========================================================
# 8) Construcción del grafo
#    - OverallState: estado interno completo
#    - input_schema: esquema permitido al invocar
#    - output_schema: esquema devuelto al finalizar
# =========================================================
builder = StateGraph(
    OverallState,
    input_schema=InputState,
    output_schema=OutputState,
)

builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", "node_3")
builder.add_edge("node_3", END)

graph = builder.compile()

# =========================================================
# 9) Ejecutar el grafo
# =========================================================
result = graph.invoke({"user_input": "My"})

print(result)
# {'graph_output': 'My name is Lance'}















