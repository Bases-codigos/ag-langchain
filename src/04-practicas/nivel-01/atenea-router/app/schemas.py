from typing_extensions import Literal, TypedDict, List
from pydantic import BaseModel, Field

AgentMode = Literal["general", "technical", "memory_update"]

class UserMessage(BaseModel):
    user_id: str = Field(..., description="Identificación única del usuario")
    thread_id: str = Field(..., description="Identificación única de la conversación")
    message: str = Field(..., description="Mensaje del usuario")

class RoutingDecision(BaseModel):
    mode: AgentMode
    model_name: str
    system_prompt_key: str
    tool_names: List[str]

class AgentResponse(BaseModel):
    answer: str
    used_model: str

class ShortTermTurn(TypedDict):
    role: str
    content: str