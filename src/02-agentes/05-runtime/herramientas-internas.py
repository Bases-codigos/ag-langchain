from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Context:
    user_id: str

def fetch_user_email_preferences(runtime: ToolRuntime[Context]):
    """Obtiene las preferencias de correo del usuario desde el Store"""
    user_id = runtime.context.user_id

    preferences: str = "El usuario prefiere que escribas un correo breve y educado."

    if runtime.store:
        if memory := runtime.store.get(("users",), user_id):
            preferences = memory.value["preferences"]

    print(f"\n[🔧 HERRAMIENTA] -> Preferencias obtenidas para {user_id}: '{preferences}'")
    return preferences

store = InMemoryStore()

store.put(
    ("users",), "user_123", {"preferences": "Escribe en un tono altamente formal, académico"}
)

store.put(
    ("users",), "user_456", {"preferences": "Usa mucha jerga moderna, emojis y manda mensaje"}
)

agent = create_agent(
    model="gpt-4o-mini", # O el modelo que estés utilizando
    tools=[fetch_user_email_preferences],
    context_schema=Context,
    store=store, # Pasamos el store para que ToolRuntime pueda acceder a él
    system_prompt=(
        "Eres un asistente de redacción de correos. "
        "Antes de escribir CUALQUIER correo, DEBES usar la herramienta "
        "'fetch_user_email_preferences' para saber cómo quiere el usuario que escribas."
    )
)

if __name__ == "__main__":
    prompt = "Redacta un correo a Recursos Humanos solicitando mis vacaciones para la primera semana de agosto."

    print("--- Prueba con USER_123 (Formal) ---")
    context_123 = Context(user_id="user_123")
    response_formal = agent.invoke(
        {"messages": [("user", prompt)]}, #type:ignore
        context=context_123
    )
    print(f"\n🤖 Respuesta:\n{response_formal['messages'][-1].content}")

    print("\n" + "="*50)

    print("\n--- Prueba con USER_456 (Informal) ---")
    context_456 = Context(user_id="user_456")
    response_informal = agent.invoke(
        {"messages": [("user", prompt)]}, #type:ignore
        context=context_456
    )
    print(f"\n🤖 Respuesta:\n{response_informal['messages'][-1].content}")

    print("\n" + "="*50)

    print("\n--- Prueba con USUARIO DESCONOCIDO (Default) ---")
    # Este usuario no está en el Store, así que usará el default "breve y educado"
    context_unknown = Context(user_id="user_999")
    response_default = agent.invoke(
        {"messages": [("user", prompt)]}, #type:ignore
        context=context_unknown
    )
    print(f"\n🤖 Respuesta:\n{response_default['messages'][-1].content}")



