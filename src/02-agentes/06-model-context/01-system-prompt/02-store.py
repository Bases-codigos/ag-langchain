from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langgraph.store.memory import InMemoryStore
from dotenv import load_dotenv
from pydantic_core.core_schema import none_schema

load_dotenv()

@dataclass
class Context:
    user_id: str

@dynamic_prompt
def store_aware_prompt(request: ModelRequest) -> str:
    """Modifica el System Prompt leyendo las preferencias guardadas en el Store"""
    user_id = request.runtime.context.user_id

    store = request.runtime.store

    user_prefs = None
    if store:
        user_prefs = store.get(("preferences",), user_id)

    base = "Eres un asistente útil y amigable"

    if user_prefs:
        style = user_prefs.value.get("communication_style")
        base += f"\nEl usuario prefiere respuestas con un estilo: {style}"
    else:
        base += f"\nEl usuario no tiene preferencias"

    return base

mem_store = InMemoryStore()

mem_store.put(
    ("preferences",),
    "user_123",
    {"communication_style": "extremadamente directo"}
)

mem_store.put(
    ("preferences",),
    "user_456",
    {"communication_style": "muy poético, usando metáforas de la naturaleza y bastante detallado"}
)

agent = create_agent(
    model="gpt-4o-mini", # Usamos un modelo válido
    tools=[],            # Lista vacía ya que no necesitamos herramientas para este ejemplo
    middleware=[store_aware_prompt], # type: ignore
    context_schema=Context,
    store=mem_store      # Pasamos el store que ya poblamos con datos
)

if __name__ == "__main__":
    prompt_usuario = "Explícame qué es el agua."

    print("--- Prueba con USER_123 (Estilo Directo) ---")
    response_123 = agent.invoke(
        {"messages": [{"role": "user", "content": prompt_usuario}]},
        context=Context(user_id="user_123")
    )
    print(response_123["messages"][-1].content)

    print("\n" + "="*50 + "\n")

    print("--- Prueba con USER_456 (Estilo Poético) ---")
    response_456 = agent.invoke(
        {"messages": [{"role": "user", "content": prompt_usuario}]},
        context=Context(user_id="user_456")
    )
    print(response_456["messages"][-1].content)

    print("\n" + "="*50 + "\n")

    print("--- Prueba con NUEVO USUARIO (Sin preferencias) ---")
    response_new = agent.invoke(
        {"messages": [{"role": "user", "content": prompt_usuario}]},
        context=Context(user_id="user_999")
    )
    print(response_new["messages"][-1].content)
