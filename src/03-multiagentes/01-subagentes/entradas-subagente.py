from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

# 1. Definición del estado
class DevOpsAgent(AgentState):
    server_ip: str
    environment: str

# 2. Subagente Especializado
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

subagent_db = create_agent(
    model = llm,
    tools = [],
    system_prompt = SystemMessage(
        content = (
            "Eres un ingeniero experto en bases de datos PostgreSQL"
            "Tu trabajo es leer síntomas y el entorno, y dar un diagnóstico técnico de lo que podría estar fallando"
        )
    ),
)

# 3. La herramienta Puente
@tool(
    "check_database_logs",
    description = "Útil para revisar problemas de base de datos. Pásale una descripción del problema (query)."
)
def check_database_logs(query: str, runtime: ToolRuntime[None, DevOpsAgent]):
    ip = runtime.state["server_ip"]
    entorno = runtime.state["environment"]

    print("\n[Log Interno] Coordinador solicitó revisión de DB.")
    print(f"[Log Interno] Subagente conectándose a IP: {ip} en entorno: {entorno}...")

    # construimos un prompt limpio y específico solo con lo que el subagente necesita saber.
    prompt_limpio = f"Entorno: {entorno}\nIP del servidor: {ip}\nSíntoma reportado: {query}"

    result = subagent_db.invoke({
        "messages": [HumanMessage(content=prompt_limpio)],
        # Pasamos las variables requeridas para que el agente no falle
        "server_ip": ip,
        "environment": entorno
    })

    # Devolvemos el diagnóstico del coordinador
    return result["messages"][-1].content

# 4. Agente Principal (Coordinador)
coordinador_agent = create_agent(
    model = llm,
    tools = [check_database_logs],
    state_schema = DevOpsAgent,
    system_prompt = SystemMessage(
        content = (
            "Eres el coordinador de DevOps, Un usuario reportará un problema con la aplicación."
            "Si el problema parece estar en la base de datos, usa la herramienta 'check_database_logs'."
        )
    )
)

print("--- Iniciando Monitoreo DevOps ---")

# Simulamos una alerta que llega del sistema
estado_inicial = {
    "messages": [HumanMessage(content="¡Alerta! Los usuarios dicen que la API está dando Timeout y no pueden guardar registros.")],
    "server_ip": "192.168.1.105",
    "environment": "PRODUCCIÓN"
}

# Ejecutamos el orquestador
respuesta = coordinador_agent.invoke(estado_inicial)

print("\n--- Respuesta Final del Coordinador al Usuario ---")
print(respuesta["messages"][-1].content)

