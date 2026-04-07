from dataclasses import dataclass
from typing import Callable
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import wrap_model_call,ModelRequest, ModelResponse
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)

@dataclass
class Context:
    user_id: str

@tool
def public_search(query: str):
    """Herramienta de información de usuarios"""
    return "pepito es un trabajador de onempresas"

@tool
def private_search(query: str):
    """Herramienta de consulta de usuarios"""
    return "El usuario trabaja de 8am a 5pm"

@wrap_model_call
def state_based_tools(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    state = request.state
    messages = state.get("messages", [])
    user_message_count = sum(
        1 for m in messages if getattr(m, "type", None) == "human"
    )

    tools = request.tools or []

    if user_message_count < 2:
        tools = [t for t in tools if t.name.startswith("public")]
        request = request.override(tools=tools)

    selected_names = [t.name for t in request.tools]
    print(f"Herramientas disponibles: {selected_names}")
    return handler(request)

checkpoint = InMemorySaver()

# Crear el agente
agent = create_agent(
    model=llm,
    tools = [public_search, private_search],
    system_prompt="Eres un asistente útil, responde únicamente con la información de las tool",
    middleware=[state_based_tools],
    context_schema=Context,
    checkpointer=checkpoint
)

if __name__ == "__main__":
    response = agent.invoke(
        {"messages": HumanMessage("Donde trabaja pepito?")},
        config = {"configurable": {"thread_id": "1"}}
    )

    print(response["messages"][-1].content)

    response = agent.invoke(
        {"messages": HumanMessage("Cual es el horario del usuario?")},
        config={"configurable": {"thread_id": "1"}}
    )

    print(response["messages"][-1].content)
