from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Context:
    user_role: str

@tool
def web_search(query: str) -> str:
    """Busca información actualizada en la web."""
    print(f"\n[🌐 BÚSQUEDA WEB EJECUTADA] -> Buscando: '{query}'")
    # Simulamos un extracto genérico de internet para el ejemplo
    return (
        "Machine learning (ML) es un campo de la inteligencia artificial que "
        "utiliza algoritmos y datos para imitar la forma en que los humanos aprenden, "
        "mejorando gradualmente su precisión a través de redes neuronales, árboles de decisión, etc."
    )

@dynamic_prompt
def user_role_prompt(
        request: ModelRequest
) -> str:
    """Genera system_prompt basado en el rol del usuario"""

    if request.runtime is None or request.runtime.context is None:
        user_role = "user"
    else:
        user_role = request.runtime.context.user_role

    base_prompt = "Eres un asistente útil"

    if user_role == "beginner":
        return f"{base_prompt}\nProporciona una respuesta corta"
    elif user_role == "expert":
        return f"{base_prompt}\nProporciona una respuesta matematica"
    else:
        return f"{base_prompt}"

agent = create_agent(
    model="gpt-4o-mini",  # Usamos un modelo válido y rápido
    tools=[web_search],
    middleware=[user_role_prompt], #type: ignore
    context_schema=Context
)

# 5. Ejecutar el Agente para comparar los resultados
if __name__ == "__main__":
    prompt = "Explícame qué es el Machine Learning"

    print("--- Prueba con ROL: BEGINNER (Principiante) ---")
    # El modelo actuará como un maestro de primaria
    result_beginner = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        context=Context(user_role="beginner") #type: ignore
    )
    # Extraemos el contenido del último mensaje (la respuesta del asistente)
    print(result_beginner["messages"][-1].content)

    print("\n" + "="*50 + "\n")

    print("--- Prueba con ROL: EXPERT (Experto) ---")
    # El modelo actuará como un ingeniero hablando con otro ingeniero
    result_expert = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        context=Context(user_role="expert") #type: ignore
    )
    print(result_expert["messages"][-1].content)





