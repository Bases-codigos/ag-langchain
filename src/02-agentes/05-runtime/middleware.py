from dataclasses import dataclass
from typing import Any
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import dynamic_prompt,ModelRequest, before_model, after_model
from langgraph.runtime import Runtime
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Context:
    user_name: str

@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    """Configura el system prompt inyectando el nombre del usuario"""
    if request.runtime is None or request.runtime.context:
        user_name = "user"
    else:
        user_name = request.runtime.context.user_name

    return f"Eres un asistente útil y amigable. Dirigete al usuario por su nombre: {user_name}"

@before_model
def log_before_model(state: AgentState, runtime: Runtime[Context]) -> dict | None:
    print(f"\n[⏳ BEFORE HOOK] -> Procesando la solicitud para el usuario: {runtime.context.user_name}")

@after_model
def log_after_model(state: AgentState, runtime: Runtime[Context]) -> dict[str, Any] | None:
    print(f"[✅ AFTER] Solicitud completada para el usuario: {runtime.context.user_name}")

agent = create_agent(
    model="gpt-4o-mini",  # Actualizado a un modelo válido
    tools=[],             # Corregido: Lista vacía en lugar de [...]
    middleware=[dynamic_system_prompt, log_before_model, log_after_model], # type: ignore
    context_schema=Context
)

if __name__ == "__main__":
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "¿Cuál es mi nombre?"}]},
        context=Context(user_name="John Smith")
    )
    print(f"🤖 Agente: {response['messages'][-1].content}")