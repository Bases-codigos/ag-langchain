from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Context:
    user_name: str

@dynamic_prompt
def inject_prompt_name(request: ModelRequest) -> str:
    """Extrae el nombre del contexto y lo inyecta al system prompt"""
    if request.runtime is None or request.runtime.context is None:
        name = "Usuario desconocido"
    else:
        name = request.runtime.context.user_name

    return f"Eres un asistente amigable y directo. El nombre del usuario es {name}"

agent = create_agent(
    model="gpt-4.1-mini",
    tools=[],
    middleware=[inject_prompt_name],
    context_schema=Context,
)

if __name__ == "__main__":
    response = agent.invoke(
        {"messages": {"role": "user", "content": "¿Cual es mi nombre?"}},
        context=Context(user_name="Harold"),
    )
    print(response["messages"][-1].content)

