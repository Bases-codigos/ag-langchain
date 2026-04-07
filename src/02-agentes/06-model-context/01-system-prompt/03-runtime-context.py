from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Context:
    user_role: str
    deployment_env: str

@dynamic_prompt
def context_aware_prompt(request: ModelRequest) -> str:
    """Extrae el nombre del contexto y lo inyecta al system prompt"""
    user_role = request.runtime.context.user_role
    env = request.runtime.context.deployment_env

    base = "Eres un asistente útil"

    if user_role == "admin":
        base += f"\nTienes acceso de administrador. Puedes realizar todas las operaciones"
    elif user_role == "viewer":
        base += f"\nSolo tienes acceso a lectura. Guía a los usuarios para que lean solo operaciones"

    if env == "production":
        base += "\nTen mucho cuidado con cualquier modificación de datos"

    return base

agent = create_agent(
    model = "gpt-4.1-mini",
    tools = [],
    middleware = [context_aware_prompt],
    context_schema = Context
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "Que permisos tengo?"}]},
    context = Context(user_role="admin", deployment_env="production")
)

print(response["messages"][-1].content)