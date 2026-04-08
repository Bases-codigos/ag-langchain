from collections import defaultdict
from typing import Dict, List

class LongTermMemory:
    """
    Memoria persistente simple basada en hechos del usuario.
    En producción esto podría vivir en Postgres, Redis o un vector store.
    """
    def __init__(self) -> None:
        self._facts: Dict[str, List[str]] = defaultdict(list)

    def add_fact(self, user_id: str, fact: str) -> None:
        self._facts[user_id].append(fact)

    def get_facts(self, user_id: str) -> List[str]:
        return self._facts.get(user_id, [])