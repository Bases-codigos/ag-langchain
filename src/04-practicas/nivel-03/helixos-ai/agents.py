from langchain.agents import create_agent
from context import AppContext
from memory import store
from middleware import helix_dynamic_system_prompt, select_model_and_tools
from tools import ALL_TOOLS
from dotenv import load_dotenv

load_dotenv()

COMMON_MIDDLEWARE = [
    helix_dynamic_system_prompt,
    select_model_and_tools,
]

billing_agent = create_agent(
    model = "gpt-4.1-mini",
    tools = ALL_TOOLS,
    middleware = COMMON_MIDDLEWARE,
    context_schema = AppContext,
    store = store,
)

incidents_agent = create_agent(
    model = "gpt-4.1-mini",
    tools = ALL_TOOLS,
    middleware = COMMON_MIDDLEWARE,
    context_schema = AppContext,
    store = store,
)

research_agent = create_agent(
    model = "gpt-4.1-mini",
    tools = ALL_TOOLS,
    middleware = COMMON_MIDDLEWARE,
    context_schema = AppContext,
    store = store,
)