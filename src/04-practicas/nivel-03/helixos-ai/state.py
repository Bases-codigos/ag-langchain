# Define estado compartido del grafo
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class HelixState(TypedDict, total=False):
    user_message: str
    route: Literal["billing", "incident", "research", "mixed", "clarify"]
    requires_human_approval: bool

    billing_result: str
    incident_result: str
    research_result: str
    final_response: str

    active_skills: list[str]

    # Reducers: agregan evidencia producida por distintos nodos/subagentes
    notes: Annotated[list[str], add_messages]
    actions_taken: Annotated[list[str], add_messages]
    escalations: Annotated[list[str], add_messages]

# Se usa reducers para campos donde varios nodos pueden escribir sin pisarse
# Esa es la idea natural para listas de evidencia, acciones o hallazgos en
# flujos paralelos o multi-steps. En GraphAPI, el state es el snapshot compartido
# del workflow y los reducers definen cómo se fusionan actualizaciones concurrentes
