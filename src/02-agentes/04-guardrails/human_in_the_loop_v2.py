from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. Definición de las Herramientas
# ==========================================

@tool
def search_tool(query: str) -> str:
    """Busca información en la base de datos de la empresa."""
    return f"Resultados para '{query}': No se encontraron anomalías."


@tool
def send_email_tool(to: str, body: str) -> str:
    """Envía un correo electrónico al destinatario especificado."""
    # En un entorno real, aquí te conectarías a un servidor SMTP o API (ej. SendGrid)
    return f"Éxito: Correo enviado a {to} con el mensaje: '{body}'"


@tool
def delete_database_tool(db_name: str) -> str:
    """Elimina la base de datos especificada. ¡Acción destructiva!"""
    return f"Éxito: Base de datos '{db_name}' eliminada permanentemente."


# ==========================================
# 2. Configuración del Modelo y Agente
# ==========================================

# Usamos un modelo real, configurando la temperatura en 0 para mayor precisión
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Persistencia en memoria RAM
checkpointer = InMemorySaver()

agent = create_agent(
    model=llm,
    tools=[search_tool, send_email_tool, delete_database_tool],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                # ¡Importante! Las claves deben coincidir con el nombre exacto de la función @tool
                "send_email_tool": True,
                "delete_database_tool": True,
                "search_tool": False,
            }
        ),
    ],
    checkpointer=checkpointer,
)

# ==========================================
# 3. Flujo de Ejecución (El Human-in-the-Loop)
# ==========================================

# Se requiere un ID de hilo para que el checkpointer sepa qué conversación rastrear
config = {"configurable": {"thread_id": "hilo_produccion_001"}}

print("--- FASE 1: El usuario da la orden ---")
orden = "Envia un correo al grupo team@example.com diciendo 'Hello everyone!'"
print(f"Usuario: {orden}\n")

# El agente procesará la orden, decidirá usar 'send_email_tool' y se pausará automáticamente.
result = agent.invoke(
    {"messages": [{"role": "user", "content": orden}]},
    config=config
)

# Comprobamos si el agente está esperando (pausado)
estado = agent.get_state(config)
if estado.next:
    print("--- FASE 2: AGENTE PAUSADO (Esperando intervención humana) ---")

    # Extraemos qué herramienta intentaba usar el modelo y con qué argumentos
    tarea_pendiente = estado.tasks[0].interrupts[0].value
    herramienta_solicitada = tarea_pendiente.get("action_requests")[0]["name"]
    argumentos_propuestos = tarea_pendiente.get("action_requests")[0]["args"]

    print(f"Alerta: El agente quiere usar la herramienta: '{herramienta_solicitada}'")
    print(f"Argumentos propuestos por la IA: {argumentos_propuestos}\n")

    print("--- FASE 3: Intervención del Humano (Edición) ---")
    print("El humano decide corregir el mensaje antes de enviarlo...")

    # En lugar de solo aprobar ("type": "approve"), usamos "edit" para modificar el parámetro 'body'
    comando_decision = Command(
        resume={
            "decisions": [
                {
                    "type": "edit",
                    "edited_action": {
                        "name": herramienta_solicitada,
                        # Mantenemos el destinatario original, pero mejoramos el texto
                        "args": {
                            "to": argumentos_propuestos["to"],
                            "body": "Hello everyone! Esta es el Mensaje Oficial."
                        }
                    }
                }
            ]
        }
    )

    # Reanudamos el agente con nuestra edición inyectada
    result_final = agent.invoke(comando_decision, config=config)

    print("\n--- FASE 4: Respuesta Final ---")
    print(result_final["messages"][-1].content)