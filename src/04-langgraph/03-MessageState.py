from langgraph.graph import StateGraph, START, END, MessagesState
from langchain.messages import HumanMessage, AIMessage

class State(MessagesState):
    documents: list[str]

def buscar_documentos(state: State):
    ultimo_mensaje = state["messages"][-1].content

    docs = [
        f"Documento relacionado con: {ultimo_mensaje}",
        "Otro documento de apoyo"
    ]

    return {
        "documents": docs,
    }

def responder(state: State) -> dict:
    pregunta = state["messages"][-1].content
    docs = state.get("documents", [])

    return {
        "messages": [
            AIMessage(
                content=f"Tu pregunta fue: {pregunta}\n"
                f"Encontré {len(docs)} documentos"
            )
        ]
    }

builder = StateGraph(State)
builder.add_node("buscar_documentos", buscar_documentos)
builder.add_node("responder", responder)

builder.add_edge(START, "buscar_documentos")
builder.add_edge("buscar_documentos", "responder")
builder.add_edge("responder", END)

graph = builder.compile()

result = graph.invoke(
    {
        "messages": [HumanMessage(content="Hablame de Kubernetes")],
        "documents": []
    }
)

for msg in result["messages"]:
    msg.pretty_print()

print(f"Resultado de documents: {result["documents"]}")