from dataclasses import dataclass
from typing import Callable
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelResponse, ModelRequest
from dotenv import load_dotenv
from langgraph.store.memory import InMemoryStore

load_dotenv()

# 1. Esquema del Contexto
@dataclass
class Context:
    user_id: str

# 2. Middleware: Inyector de Estilo
@wrap_model_call
def inject_writing_style(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Inyecta el estilo de redacción de correo electrónico del usuario desde el Store"""
    user_id = request.runtime.context.user_id if request.runtime and request.runtime.context else None

    store = request.runtime.store

    if not store:
        return handler(request)

    writing_style = store.get(("writing_style", user_id))

    if writing_style:
        style = writing_style.value
        style_context = f"""Reglas de estilo obligatorias para este correo:
- Tono: {style.get("tone", "professional")}
- Saludo típico: "{style.get("greeting", "Hi")}"
- Despedida típica: "{style.get("sign-off", "Best")}"
- Ejemplo de correo electrónico que has escrito:
{style.get('example_email', '')}
"""
        messages = [
            *request.messages,
            {"role": "system", "content": style_context},
        ]

        request = request.override(messages=messages)

    return handler(request)

mem_store = InMemoryStore()

mem_store.put(
    ("writing_style",),
    "user_formal",
    {
        "tone": "estrictamente formal, corporativo y directo",
        "greeting": "Estimado/a,",
        "sign_off": "Atentamente,",
        "example_email": "Adjunto el reporte correspondiente al Q3 para su revisión. Quedo a su disposición para cualquier duda."
    }
)

# Perfil 2: Usuario relajado
mem_store.put(
    ("writing_style",),
    "user_informal",
    {
        "tone": "muy relajado, amigable, entusiasta y usando emojis",
        "greeting": "¡Hola equipo!",
        "sign_off": "¡Un abrazo grande!",
        "example_email": "¡Quería contarles que ya terminamos la fase 1! 🚀 Gran trabajo de todos, ¡vamos por más!"
    }
)

agent = create_agent(
    model="gpt-4o-mini",
    tools=[], # Lista vacía ya que la lógica depende totalmente del store y el prompt
    middleware=[inject_writing_style],
    context_schema=Context,
    store=mem_store # Le pasamos el store poblado
)

if __name__ == "__main__":
    prompt = "Escribe un correo avisando que la reunión de mañana se pospone al viernes a las 10 AM."

    print("=== Prueba con USUARIO FORMAL ===")
    response_formal = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        context=Context(user_id="user_formal")
    )
    print(f"{response_formal['messages'][-1].content}\n")

    print("=== Prueba con USUARIO INFORMAL ===")
    response_informal = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        context=Context(user_id="user_informal")
    )
    print(f"{response_informal['messages'][-1].content}\n")















