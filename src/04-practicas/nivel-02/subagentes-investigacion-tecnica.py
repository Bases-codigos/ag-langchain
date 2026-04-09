from typing import Any
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import AIMessage
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 1) SUBAGENTE ESPECIALISTA EN INVESTIGACIÓN TÉCNICA
# ============================================================
research_agent = create_agent(
    model="openai:gpt-4.1",
    tools=[],
    system_prompt=(
        "Eres un subagente especialista en investigación técnica. "
        "Tu trabajo es analizar problemas de software, identificar causas, "
        "comparar opciones y devolver respuestas precisas y estructuradas. "
        "No hables como asistente generalista. No saludes. No inventes datos."
    ),
)

@tool("technical_research")
def technical_research(query: str) -> str:
    """
    Investiga un problema técnico y devuelve hallazgos concretos.
    """
    result: dict[str, Any] = research_agent.invoke(
        {
            "messages": {
                "role": "user",
                "content": query,
            }
        }
    )

    final_message = result["messages"][-1]
    if isinstance(final_message, AIMessage):
        return str(final_message.content)

    return str(final_message)

# ============================================================
# 2) SUBAGENTE ESPECIALISTA EN REDACCIÓN
# ============================================================
writer_agent= create_agent(
    model="gpt-4.1",
    tools=[],
    system_prompt=(
        "Eres un subagente especialista en redacción técnica profesional. "
        "Transformas contenido técnico en texto claro, breve y útil para correos, "
        "resúmenes ejecutivos, reportes o documentación. "
        "Mantén tono profesional y preciso."
    )
)

@tool("technical_writer")
def technical_writer(draft_request: str) -> str:
    """
    Redacta o mejora contenido técnico con claridad profesional.
    """
    result: dict[str, Any] = writer_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": draft_request,
                }
            ]
        }
    )

    final_message = result["messages"][-1]
    if isinstance(final_message, AIMessage):
        return str(final_message.content)

    return str(final_message)

# ============================================================
# 3) AGENTE SUPERVISOR
# ============================================================
supervisor_agent = create_agent(
    model="openai:gpt-4.1",
    tools=[technical_research, technical_writer],
    system_prompt=(
        "Eres un supervisor multiagente para soporte técnico interno. "
        "Debes decidir cuándo usar technical_research y cuándo usar "
        "technical_writer. "
        "Usa technical_research para analizar problemas, arquitecturas, causas "
        "raíz o comparar alternativas. "
        "Usa technical_writer para reescribir, resumir, documentar o preparar "
        "mensajes profesionales. "
        "Si hace falta, puedes usar primero technical_research y luego "
        "technical_writer para producir una respuesta final mejor."
    ),
)

# ============================================================
# 4) FUNCIÓN DE EJECUCIÓN
# ============================================================

def run_supervisor(user_input: str) -> str:
    """
    Ejecuta el supervisor y devuelve el último mensaje de salida.
    """
    result: dict[str, Any] = supervisor_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_input,
                }
            ]
        }
    )

    final_message = result["messages"][-1]
    return str(final_message.content)

if __name__ == "__main__":
    question = (
        "Tengo un error intermitente en FastAPI al usar un grafo de LangGraph. "
        "Primero analiza posibles causas y luego redacta una respuesta corta "
        "para enviarla al equipo de backend."
    )
    output = run_supervisor(question)
    print(output)