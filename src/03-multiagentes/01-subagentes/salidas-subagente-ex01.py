from typing import Annotated
from pydantic import BaseModel, Field
from langchain.agents import create_agent, AgentState
from langchain.tools import tool, InjectedToolCallId
from langgraph.types import Command
from langchain.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. Definición del Estado y Esquemas (Pydantic)
# ==========================================
class SupportState(AgentState):
    escalation_needed: bool # Estado global que el subagente modificará

class DiagnosticResult(BaseModel):
    """Obligamos al subagente a responder con este formato estricto"""
    analisis: str = Field(description="Una explicación técnica breve del problema reportado.")
    requiere_humano: bool = Field(description="True si el problema es crítico (ej. caída de servidor, pérdida de datos. False si es una consulta simple (ej. reseteo de contraseña).")

# ==========================================
# 2. Subagente (Con Salida Estructurada)
# ==========================================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# En lugar de un agente completo, usamos with_structured_output, nos devuelve un objeto Python (DiagnosticResult), no un string.
subagent_analyzer = llm.with_structured_output(DiagnosticResult)

# ==========================================
# 3. Herramientas del Orquestador
# ==========================================
@tool(
    "diagnose_issue",
    description="Usa esta herramienta PRIMERO para analizar cualquier problema técnico reportado por usuario."
)
def diagnose_issue(query: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    print(f"\n[Herramienta] Subagente analizando: '{query}'...")

    # 1. Ejecutamos el subagente. Al usar with_structured_output, nos devuelve un objeto Python (DiagnosticResult), no un string.
    resultado: DiagnosticResult = subagent_analyzer.invoke([
        SystemMessage(content="Eres un ingeniero de soporte Nivel 2. Clasifica la gravedad del problema."),
        HumanMessage(content=query)
    ])

    print(f"[Herramienta] Análisis completo. ¿Requiere escalar a un humano?: {resultado.requiere_humano}")

    # 2. Formateamos el texto que leerá el Agente Principal
    texto_para_llm = (
        f"Diagnóstico Técnico: {resultado.analisis}\n"
        f"ESTADO DE ESCALAMIENTO: {'REQUERIDO' if resultado.requiere_humano == True else 'NO REQUERIDO'}"

    )

    # 3. Retornamos el Command alterando el estado y satisfaciendo el tool_call
    return Command(
        update={
            "escalation_needed": resultado.requiere_humano, # Actualizamos la variable global
            "messages": [ToolMessage(content=texto_para_llm, tool_call_id=tool_call_id)],
        }
    )

@tool(
    "escalate_to_human",
    description="Usa esta herramienta SOLAMENTE si el diagnóstico indicó que el ESTADO DE ESCALAMIENTO es REQUERIDO."
)
def escalate_to_human(resume_problema: str) -> str:
    # Aquí iría el código real para enviar un mensaje o crear un ticket para el especialista
    print(f"[ALERTA SISTEMA] ¡Ticket enviado al equipo de ingenieros de guardia!")
    print(f"[ALERTA SISTEMA] Detalles: {resume_problema}")

    return "Éxito: El equipo humano ha sido notificado!"

# ==========================================
# 4. Agente Principal (Orquestador L1)
# ==========================================
# El System Prompt es la clave para el enrutamiento condicional
prompt_orquestador = """
Eres el primer contacto de soporte técnico. Sigue estos pasos ESTRICTAMENTE:
1. Usa la herramienta 'diagnose_issue' para analizar el problema del usuario.
2. Lee el resultado de la herramienta.
3. Si la herramienta dice que el ESTADO DE ESCALAMIENTO es REQUERIDO, DEBES usar la herramienta 'escalate_to_human' inmediatamente.
4. Si NO ES REQUERIDO, simplemente ofrécele una solución básica al usuario.
"""

main_agent = create_agent(
    model = llm,
    tools = [diagnose_issue, escalate_to_human],
    state_schema = SupportState,
    system_prompt = SystemMessage(content=prompt_orquestador),
)

print("--- PRUEBA 1: Problema Grave (Debería usar ambas herramientas) ---")
estado_grave = {
    "messages": [HumanMessage(content="¡Ayuda! La base de datos dfe producción se acaba de borrar accidentalmente y el servidor no response.")],
    "escalation_needed": False,
}

respuesta_grave = main_agent.invoke(estado_grave)
print(f"Respuesta L1: {respuesta_grave["messages"][-1].content}")

print("\n\n--- PRUEBA 2: Problema Simple (Debería usar solo la primera herramienta) ---")
estado_simple = {
    "messages": [HumanMessage(content="Hola, olvidé mi contraseña del portal de nómina, ¿cómo la recupero?")],
    "escalation_needed": False,
}

respuesta_simple = main_agent.invoke(estado_simple)
print(f"Respuesta L1: {respuesta_simple['messages'][-1].content}")













