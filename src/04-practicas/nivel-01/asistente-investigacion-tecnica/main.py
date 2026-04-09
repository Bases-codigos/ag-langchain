from codecs import namereplace_errors
from dataclasses import dataclass
from typing import Callable

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, wrap_model_call, ModelRequest, ModelResponse
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from dotenv import load_dotenv

load_dotenv()

# =========================================================
# 1) Contexto de runtime
# =========================================================
@dataclass
class ResearchContext:
    user_id: str

# =========================================================
# 2) Infraestructura de memoria
# =========================================================
# Short-term memory: por hilo
checkpointer = InMemorySaver()

# Long-term memory: entre hilos
store = InMemoryStore()

# =========================================================
# 3) Modelos
# =========================================================
fast_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
smart_model = ChatOpenAI(model="gpt-4.1", temperature=0.1)

# =========================================================
# 4) Tools de memori larga
# =========================================================
@tool
def save_research_interest(
        topic: str,
        runtime: ToolRuntime[ResearchContext],
) -> str:
    """
    Guarda un área de interés persistente del usuario.
    Ejemplo: 'LangGraph', 'RAG', 'MCP', 'observabilidad de agentes'
    """
    assert runtime.context.user_id

    user_id = runtime.context.user_id
    namespace = ("research_profile",)
    key = user_id

    existing = runtime.store.get(namespace, key)

    if existing and isinstance(existing.value, dict):
        interests = list(existing.value.get("interests", []))
        style_preferences = list(existing.value.get("style_preferences", []))
    else:
        interests = []
        style_preferences = []

    if topic not in interests:
        interests.append(topic)

    runtime.store.put(namespace, key, {"interests": interests, "style_preferences": style_preferences})

    return f"Interés guardado: {topic}"

@tool
def save_style_preference(
        preference: str,
        runtime: ToolRuntime[ResearchContext],
) -> str:
    """
    Guarda una preferencia de estilo persistente.
    Ejemplo: 'prefiero respuestas con trade-offs'
    """
    assert runtime.store is not None

    user_id = runtime.context.user_id
    namespace = ("research_profile",)
    key = user_id

    existing = runtime.store.get(namespace, key)

    if existing and isinstance(existing.value, dict):
        interests = list(existing.value.get("interests", []))
        style_preferences = list(existing.value.get("style_preferences", []))
    else:
        interests = []
        style_preferences = []

    if preference not in interests:
        style_preferences.append(preference)

    runtime.store.put(namespace, key, {"style_preferences": style_preferences})

    return f"Preferencia guardada: {preference}"

@tool
def get_research_profile(runtime: ToolRuntime[ResearchContext]) -> str:
    """
    Recupera el perfil persistente del usuario.
    """
    assert runtime.store is not None

    user_id = runtime.context.user_id
    profile = runtime.store.get(("research_profile",), user_id)

    if not profile:
        "No hay perfil persistente guardado."

    interests = profile.value.get("interests", [])
    style_preferences = profile.value.get("style_preferences", [])

    lines = []

    if interests:
        lines.append("Intereses:")
        lines.extend(f"- {item}" for item in interests)

    if style_preferences:
        lines.append("Preferencias de estilo:")
        lines.extend(f"- {item}" for  item in style_preferences)

    if not lines:
        return "No hay perfil persistente guardado."

    return "\n".join(lines)

# =========================================================
# 5) Tool de conocimiento técnico
# =========================================================
@tool
def research_kb_search(query: str) -> str:
    """
    Simula búsqueda en una base de conocimiento de IA aplicada.
    """
    kb = {
        "langgraph": (
            "LangGraph está orientado a agentes y workflows con estado, "
            "persistencia, streaming y control explícito de nodos y edges."
        ),
        "rag": (
            "RAG combina recuperación de contexto externo con generación, "
            "y suele requerir estrategias de chunking, ranking y grounding."
        ),
        "mcp": (
            "MCP estandariza la exposición de herramientas y recursos para que "
            "los agentes puedan consumirlos de forma interoperable."
        ),
        "observabilidad": (
            "La observabilidad de agentes suele incluir tracing, latencia, "
            "tool calls, errores, costos y evaluación de respuestas."
        ),
    }

    q = query.lower()
    for key, value in kb.items():
        if key in q:
            return value

    return "No encontré nada relevante en la base de investigación."

# =========================================================
# 6) Prompt dinámico
# =========================================================
@dynamic_prompt
def research_prompt(request: ModelRequest) -> str:
    """
    Construye el system prompt usando long-term memory.
    """

    base_prompt = (
        "Eres Atenea Research, un asistente técnico de investigación. "
        "Tu trabajo es explicar con precisión, separar hechos de inferencias "
        "y destacar trade-offs cuando existan."
    )

    user_id = request.runtime.context.user_id
    profile = request.runtime.store.get(("research_profile",), user_id)

    if profile and isinstance(profile.value, dict):
        interests = profile.value.get("interests", [])
        style_preferences = profile.value.get("style_preferences", [])

        if interests:
            base_prompt += (
                "\nÁreas de interés persistentes del usuario:"
                + ", ".join(interests)
                + "."
            )
        if style_preferences:
            base_prompt += (
                "\nPreferecnias persistenes del usuario: "
                + "; ".join(style_preferences)
                + "."
            )

    return base_prompt

# =========================================================
# 7) Selección dinámica de modelo
# =========================================================
@wrap_model_call
def select_research_model(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    """
    Política simple:
    - modelo fuerte para consultas largas o técnicas
    - modelo rápido para preguntas breves/directas
    """
    messages = request.state.get("messages", [])

    combined = " ".join(str(getattr(msg, "content", "")) for msg in messages
    ).lower()

    hard_signals = [
        "compara",
        "trade-off",
        "arquitectura",
        "rag",
        "langgraph",
        "mcp",
        "observabilidad",
        "diseño",
        "producción",
    ]

    long_query = len(combined) > 500
    complex_query = any(word in combined for word in hard_signals)

    if long_query or complex_query or len(messages) > 8:
        model = smart_model
    else:
        model = fast_model

    return handler(request.override(model=model))

# =========================================================
# 8) Creación del agente
# =========================================================
agent = create_agent(
    model=fast_model,
    tools=[
        save_research_interest,
        save_style_preference,
        get_research_profile,
        research_kb_search,
    ],
    middleware=[
        research_prompt,
        select_research_model
    ],
    checkpointer=checkpointer,
    store=store,
    context_schema=ResearchContext,
)

# =========================================================
# 9) Demo
# =========================================================
def run_demo() -> None:
    context = ResearchContext(user_id="harold")

    # Hilo 1
    config_1 = {"configurable": {"thread_id": "research-thread-001"}}

    print("\n--- TURNO 1: guardar interés ---")
    result_1 = agent.invoke(
        {"messages": [
            {
                "role": "user",
                "content": (
                    "Quiero trabajar mucho LnagGraph este mes."
                    "Guarda ese interés usando la herramienta correspondiente."
                )
            },
        ]},
        config = config_1,
        context = context,
    )
    print(result_1["messages"][-1].content)

    print("\n--- TURNO 2: guardar preferencia de estilo ---")
    result_2 = agent.invoke(
        {"messages": [{
            "role": "user",
            "content": (
                "Recuerda que prefiero respuestas con trade-offs "
                "y enfoque de producción. Guárdalo"
            )
        }]},
        config = config_1,
        context = context,
    )
    print(result_2["messages"][-1].content)

    print("\n--- TURNO 3: consulta técnica en el mismo hilo ---")
    result_3 = agent.invoke(
        {
            "messages": [{
                "role": "user",
                "content": "Explícame qué es LnagGraph y cómo difiere de un flujo simple con LangChain."
            }]
        },
        config = config_1,
        context = context,
    )
    print(result_3["messages"][-1].content)

    # Hilo 2, mismo usuario
    config_2 = {
        "configurable": {
            "thread_id": "research-thread-999",
        }
    }

    print("\n--- TURNO 4: nuevo hilo, mismo usuario ---")
    result_4 = agent.invoke(
        {
            "messages": [{
                "role": "user",
                "content": "¿Qué recuerdad sobre mis intereses y preferencias?"
            }]
        },
        config = config_2,
        context = context,
    )
    print(result_4["messages"][-1].content)

if __name__ == "__main__":
    run_demo()