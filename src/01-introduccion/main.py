from dataclasses import dataclass

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from langchain.agents.structured_output import ToolStrategy
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Context:
    user_id: str

@dataclass
class ResponseFormat:
    punny_response: str
    weather_response: str

@tool
def get_user_location(user_id: str) -> str:
    """Obtiene la ubicación predeterminada para un ID de usuario determinado"""
    locations = {"1": "Florida"}
    return locations.get(user_id, "Desconocido")

@tool
def get_weather_for_location(location: str ) -> str:
    """Obtiene el clima actual para una ubicación específica"""
    if location.lower() == "florida":
        return "Siempre es verano en florida"
    return "Nublado"

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

checkpointer = InMemorySaver()

SYSTEM_PROMPT = """
Eres un asistente meteorológico divertidísimo.
Siempre debes responder usando el formato estructurado proporcionado.
Usa las herramientas disponibles para obtener la ubicación del usuario y el clima si es necesario.

¡No olvides incluir muchos juegos de palabras en tu respuesta ingeniosa!
"""

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer
)

config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [HumanMessage("¿Qué tiempo hace fuera??")]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])


