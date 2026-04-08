from typing_extensions import Annotated, TypedDict, List
from langchain.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

# Estado del grafo
class State(TypedDict):
    # La lista de mensajes usa add_messages para acumularlos
    messages: Annotated[List[AnyMessage], add_messages]

# Node simple
def responder(state: State) -> dict:
    ultimo_mensaje = state["messages"][-1]

    return {
        "messages": [AIMessage(content=f"Hola, dijiste {ultimo_mensaje.content}!")]
    }

# Construcción del grafo
builder = StateGraph(State)
builder.add_node("responder", responder)
builder.add_edge(START, "responder")
builder.add_edge("responder", END)

graph = builder.compile()

# Invocación
result = graph.invoke(
    {
        "messages": [
            HumanMessage(content="Me llamo Harold")
        ]
    }
)

for msg in result["messages"]:
    msg.pretty_print()