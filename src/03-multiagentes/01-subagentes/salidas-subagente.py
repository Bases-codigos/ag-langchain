from typing import Annotated
from langchain.agents import AgentState, create_agent
from langchain.tools import tool, InjectedToolCallId
from langchain.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.types import Command
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. Definición del Estado
# ==========================================
class CustomState(AgentState):
    nivel_de_confianza: str # Esta es la variable (example_state_key) que queremos actualizar

# ==========================================
# 2. Subagente (El Investigador)
# ==========================================
llm = ChatOpenAI(model = "gpt-4o-mini", temperature=0)

subagent1 = create_agent(
    model = llm,
    tools = [],
    state_schema = CustomState,
    system_prompt = SystemMessage(
        content = "Eres un investigador experto. Responde a la consulta de forma concisa."
    )
)

# ==========================================
# 3. La Herramienta con InjectedToolCallId y Command
# ==========================================
@tool(
    "subagent1_name",
    description = "Delega la investigación. Pásale el tema a investigar."
)
def call_subagent1(
        query: str,
        # Inyectamos el ID exacto que generó el LLM para esta llamada a la herramienta
        tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    print(f"\n[Subagente] Recibí la orden de investigar: '{query}'")

    # 1. Ejecutamos el subagente
    # Simulamos que el subagente analizó la data y determinó que la confianza es "ALTA"
    result = subagent1.invoke({
        "messages": [HumanMessage(content=query)],
        "nivel_de_confianza": "ALTA (Confirmado por múltiples fuentes)"
    })

    texto_respuesta = result["messages"][-1].content
    nuevo_estado_confianza = result["nivel_de_confianza"]

    print(f"[Subagente] Devolviendo respuesta y actualizando estado global a: '{nuevo_estado_confianza}'...")

    # 2. La Magia: Retornamos un Command en lugar de un string
    return Command(
        update={
            # Modificamos una llave del estado global del agente principal
            "nivel_de_confianza": nuevo_estado_confianza,

            # Formateamos correctamente la respuesta para evitar error 400
            "messages": [
                ToolMessage(
                    content=texto_respuesta,
                    tool_call_id=tool_call_id # <-- Enlazamos el mensaje con el ID original
                )
            ]
        }
    )

# ==========================================
# 4. Agente Principal (Orquestador)
# ==========================================
main_agent = create_agent(
    model = llm,
    tools = [call_subagent1],
    state_schema = CustomState,
    system_prompt = SystemMessage(
        content=(
            "Eres el orquestador principal. Delega la investigación usando 'subagent1_name'."
            "Una vez que el subagente responda, finaliza la conversación agradeciendo."
        )
    )
)

print("--- Iniciando Ejecución ---")

estado_inicial = {
    "messages": [HumanMessage(content="Averigua cuál es la capital de Francia.")],
    "nivel_de_confianza": "DESCONOCIDO", #Estado original
}

# Ejecutamos el orquestador
respuesta = main_agent.invoke(estado_inicial)

print("\n--- Resultados Finales en el Estado del Orquestador ---")
print(f"Respuesta del LLM: {respuesta['messages'][-1].content}")
print(f"Estado de la variable 'nivel_de_confianza': {respuesta['nivel_de_confianza']}")
