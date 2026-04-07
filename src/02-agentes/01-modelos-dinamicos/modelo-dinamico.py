from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.agents.structured_output import ToolStrategy
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

basic_model = ChatOpenAI(model="gpt-4.1-mini")
advance_model = ChatOpenAI(model="gpt-4.1")


@dataclass
class Context:
    user_id: str


@dataclass
class ResponseFormat:
    punny_response: str
    weather_conditions: str | None = None


@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Seleccione el modelo en función de la complejidad de la conversación"""
    message_count = len(request.state["messages"])
    if message_count > 10:
        model = advance_model
    else:
        model = basic_model

    return handler(request.override(model=model))


agent = create_agent(
    model=basic_model,
    middleware=[dynamic_model_selection],
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat)
)

config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "¿Qué tiempo hace fuera??"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])