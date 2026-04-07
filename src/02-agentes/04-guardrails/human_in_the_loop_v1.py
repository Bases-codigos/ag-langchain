from langchain.tools import tool
from langgraph.types import Command, interrupt
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# 1. Definimos las herramientas y configuramos las interrupciones
@tool
def search_tool(query: str) -> str:
    """Útil para buscar información. Operación segura (no requiere aprobación)."""
    return f"Resultados de la búsqueda para: {query}"

@tool
def send_email_tool(to: str, body: str) -> str:
    """Envía un correo electrónico. ¡Requiere aprobación humana!"""
    print(f"\n[ALERTA DEL SISTEMA] El agente quiere enviar un correo a: {to}")

    # AQUÍ ESTÁ LA MAGIA: Pausamos la ejecución del agente y pasamos contexto al humano
    decision = interrupt(
        {"action": "send_email", "to": to, "body": body}
    )

    # El código se queda congelado arriba hasta que enviemos 'Command(resumen=...)'
    if decision.get("type") == "approve":
        return f"Éxito: Correo enviado a {to}."
    else:
        return "Error: El usuario denegó la operación."

@tool
def delete_database_tool(db_name: str) -> str:
    """Boarra una base de datos. ¡Requiere aprobación humana!"""
    print(f"\n[ALERTA DEL SISTEMA] El agente quiere borrar la base de datos: {db_name}")
    decision = interrupt(
        {"action": "delete_database", "db_name": db_name}
    )

    if decision.get("type") == "approve":
        return f"Éxito: Base de datos {db_name} eliminada."

    return "Error: Operación denegada."

# 2. Instancia el modelo y el agente
llm = ChatOpenAI(model = "gpt-4.1-mini", temperature=0)

# El checkpointer guarda el estado para que el agente sobreviva a la pausa
checkpointer = InMemorySaver()

agent = create_agent(
    model = llm,
    tools = [search_tool, send_email_tool, delete_database_tool],
    checkpointer = checkpointer,
)

# 3. Flujo de ejecución (El Human-in-the-Loop)

# Se requiere un thread_id para que el checkpointer sepa qué conversación pausar/reanudar
config = {"configurable": {"thread_id": "thread_123"}}

print("--- FASE 1: El usuario da la orden ---")
# El agente empezará a trabajar, se dará cuenta de que necesita usar 'send_email_tool' y se pausará
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Envia un correo a team@empresa.com diciendo 'Hello'"}]},
    config=config
)

# Podemos verificar el estado del agente para confirmar que está esperando
estado = agent.get_state(config=config)
if estado.next:
    # Obtiene la información de lo que está esperando la interrupción
    datos_pausa = estado.tasks[0].interrupts[0].value
    print(f"--- FASE 2: AGENTE PAUSADO ---")
    print(f"El agente está esperando aprobación para: {datos_pausa['action']}")

print("\n--- FASE 3: El Humano interviene")
print("Enviamos comando de aprobación...")

# Reanudamos el agente pasando la decisión
result_final = agent.invoke(
    Command(resume={"type": "approve"}),
    config=config
)

print("\n--- FASE 4: Respuesta Final del Agente ---")
print(result_final["messages"][-1].content)
