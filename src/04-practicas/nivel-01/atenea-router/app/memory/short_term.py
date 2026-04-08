from collections import defaultdict
from typing import Dict, List
from app.schemas import ShortTermTurn

class ShortTermMemory:
    """
    Memoria conversacional por thread.
    Guarda solo los últimos N mensajes para mantener contexto reciente.
    """
    def __init__(self, max_messages: int = 8) -> None:
        self.max_messages = max_messages
        self._store: Dict[str, List[ShortTermTurn]] = defaultdict(list)

    def get(self, thread_id: str, role: str, content: str) -> None:
        history = self._store[thread_id]
        history.append({"role": role, "content": content})
        self._store[thread_id] = history[-self.max_messages:]