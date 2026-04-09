from dataclasses import dataclass
from typing import Callable

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse, dynamic_prompt
from langchain.tools import ToolRuntime, tool
from langchain.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from dotenv import load_dotenv

load_dotenv()

# Contexto
@dataclass
class Context:
    user_id: str

# Memoria corto plazo
checkpointer = InMemorySaver()
# Memoria largo plazo
store = InMemoryStore()

# Modelos
fast_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
smart_model = ChatOpenAI(model="gpt-4.1", temperature=0)

# Tools
@tool
def save_user_preference(
        preference: str,
        runtime: ToolRuntime[Context],
) -> str:
    """
    Guarda preferencia persistente del usuario.
    Ejemplo: 'prefiero respuestas cortas y técnicas'
    """
    assert runtime.store is not None

    user_id = runtime.context.user_id
    namespace = ("preferences",)

    key = user_id

    existing = runtime.store.get(namespace, key)

    if existing and isinstance(existing.value, dict):
        preferences: list[str] = list(existing.value.get("preferences", []))
    else:
        preferences = []

    preferences.append(preference)

    runtime.store.put(namespace, key, {"preferences": preferences})

    return f"Preferencia guardada: {preference}"

@tool
def get_user_preferences(runtime: ToolRuntime[Context],) -> str:
    """
    Recupera preferencias persistentes del usuario.
    """
    assert runtime.store is not None

    user_id = runtime.context.user_id
    memory = runtime.store.get(namespace=("preferences",), key=user_id)

    if not memory:
        return "No hay preferencias guardadas"

    preferences = memory.value.get("preferences", [])
    if not preferences:
        return "No hay preferencias guardadas"

    preferences_user = "Preferencias del usuario: \n- "+ "\n- ".join(preferences)
    print(preferences_user)
    return preferences_user

@tool
def kb_search(query: str) -> str:
    """
    Simula una búsqueda en una base técnica interna.
    """
    kb = {
        "langgraph": (
            "LangGraph está orientado a agentes y workflows con estado"
            "persistencia, streaming y control explícito del grafo"
        ),
        "checkpointer": (
            "El checkpointer persiste el estado del hilo y permite reanudar "
            "la ejecución mediante thread_id"
        ),
        "store": (
            "EL store permite persistir memoria a largo plazo compartida entre hilos"
        )
    }

    q = query.lower()
    for key, value in kb.items():
        if key in q:
            return value

    return "No encontré nada en la base técnica"

# Prompt dinámico
@dynamic_prompt
def preferences_aware_prompt(request: ModelRequest) -> str:
    base = (
        "Eres Atena Helpdesk, un asistente técnico profesional. "
        "Sé preciso, útil y evita inventar información"
    )

    user_id = request.runtime.context.user_id
    user_memory = request.runtime.store.get(namespace=("preferences",), key=user_id)

    if user_memory and isinstance(user_memory.value, dict):
        preferences = user_memory.value.get("preferences", [])
        if preferences:
            joined = "; ".join(preferences)
            base += f"\nTen en cuenta estas preferencias persistentes del usuario: {joined}."
    return base

# Selección dinámica del modelo
@wrap_model_call
def choose_model(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    messages = request.state.get("messages", [])

    joined_text = " ".join(
        str(getattr(msg, "content", "")) for msg in messages
    ).lower()

    technical_keywords = ["arquitectura", "debug", "langgraph", "api", "error", "tool"]

    if len(messages) > 8 or any(word in joined_text for word in technical_keywords):
        model= smart_model
    else:
        model = fast_model

    request = request.override(model=model)
    return handler(request)

agent = create_agent(
    model = fast_model,
    tools = [
        save_user_preference,
        get_user_preferences,
        kb_search,
    ],
    middleware = [
        preferences_aware_prompt,
        choose_model,
    ],
    checkpointer=checkpointer,
    store=store,
    context_schema=Context
)

def run_demo() -> None:
    config = {
        "configurable": {
            "thread_id": "thread-soporte-001",
        }
    }

    context = Context(user_id="harold")

    print("\n--- TURNO 1: guardar preferencia persistente ---")
    result_1 = agent.invoke({"messages": HumanMessage(content=(
        "Recuerda esta preferencia: prefiero animales bonitos como los gatos. "
        "Usa la herramienta adecuada para guardarlo."
    ))},
        config=config,
        context=context,
    )
    print(result_1["messages"][-1].content)

    print("\n--- TURNO 2: nueva consulta en el mismo hilo ---")
    result_2 = agent.invoke({"messages": HumanMessage(content="Explícame qué es LangGraph.")},
        config=config,
        context=context,
    )
    print(result_2["messages"][-1].content)

    print("\n--- TURNO 3: otro hilo, mismo usuario ---")
    # Nota: el hilo cambia, así que cambia la short-term memory.
    # Pero la long-term memory se mantiene porque vive en el store.
    new_config = {
        "configurable": {
            "thread_id": "thread-soporte-999",
        }
    }

    result_3 = agent.invoke({"messages": HumanMessage(content="¿Qué preferencias mías recuerdas?")},
        config=new_config,
        context=context,
    )
    print(result_3["messages"][-1].content)

if __name__ == "__main__":
    run_demo()