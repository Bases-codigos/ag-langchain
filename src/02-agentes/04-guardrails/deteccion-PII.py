# (Asumiendo que las importaciones conceptuales de tu ejemplo existen)
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

# 1. Definimos las herramientas faltantes
@tool
def customer_service_tool(query: str) -> str:
    """Útil para buscar información en la base de datos de atención al cliente."""
    return "Registro encontrado. El cliente tiene un ticket abierto."

@tool
def email_tool(mensaje: str) -> str:
    """Útil para enviar notificaciones por correo al cliente."""
    return "Correo enviado con éxito."

# 2. Instanciamos el agente con tu configuración de Middleware
agent = create_agent(
    model="gpt-4o", # Usamos gpt-4o, ya que gpt-4.1 no es un modelo oficial actual
    tools=[customer_service_tool, email_tool],
    middleware=[
        # Redactar correos: Reemplaza por una etiqueta
        PIIMiddleware(
            "email",
            strategy="redact",
            apply_to_input=True,
        ),
        # Enmascarar tarjetas: Muestra solo los últimos 4 dígitos
        PIIMiddleware(
            "credit_card",
            strategy="mask",
            apply_to_input=True,
        ),
        # Bloquear API keys: Detiene la ejecución si detecta una
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="block",
            apply_to_input=True,
        ),
    ],
)

# 3. El usuario envía información sensible (PII)
entrada_usuario = "Mi email es john.doe@example.com y mi tarjeta de credito es 5105-1051-0510-5100"

print(f"1. Lo que escribió el usuario: '{entrada_usuario}'\n")

# 4. Invocamos al agente
result = agent.invoke({
    "messages": [{"role": "user", "content": entrada_usuario}]
})

print(f"3. Lo que responde el Agente: '{result['messages'][-1].content}'")