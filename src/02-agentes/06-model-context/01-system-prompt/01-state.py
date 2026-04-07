from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Context:
    user_name: str

@dynamic_prompt
def state_aware_prompt(request: ModelRequest) -> str:
    state = request.state
    messages = state.get("messages", [])
    message_count = sum(1 for m in messages if getattr(m, "type", None))
    base = "Eres un asistente útil"
    if message_count < 10:
        base = "\nEsta conversación es larga, se conciso"
    return base

agent = create_agent(
    model="gpt-4.1-mini",
    tools=[],
    middleware=[state_aware_prompt],
    context_schema=Context
)

long_history = [
    {"role": "user", "content": "Hola"},
    {"role": "assistant", "content": "Hola, que tal?"},
    {"role": "user", "content": "Necesito contexto"},
    {"role": "assistant", "content": "Te ayudo"},
    {"role": "user", "content": "Sigue"},
    {"role": "assistant", "content": "Listo"},
    {"role": "user", "content": "Que es la inteligencia artificial"},
]

response = agent.invoke(
    {"messages": long_history},
    config={"configurable": {"thread_id": "123"}},
    context=Context(user_name="jhon")

)

print(response["messages"][-1].content)
