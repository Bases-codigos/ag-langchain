from typing import Dict

class PromptManager:
    def __init__(self) -> None:
        self._prompts: Dict[str, str] = {
            "general_assistant": (
                "Eres un asistente útil, claro y profesional. "
                "Responde de forma práctica. "
                "Si existe memoria relevante del usuario, úsala para personalizar la respuesta."
            ),
            "technical_assistant": (
                "Eres un asistente técnico senior. "
                "Responde con precisión, estructura, buenas prácticas y trade_offs. "
                "Evita inventar datos. Si usas herramientas, intégralas en tu razonamiento."
            ),
            "memory_manager": (
                "Eres un asistente encargado de identificar preferencias o datos persistentes del usuario. "
                "Si el usuario expresa una preferencia estable, conviértela en un hecho corto y claro."
            )
        }

    def get(self, key: str) -> str:
        if key not in self._prompts:
            raise KeyError(f"Prompt '{key}' no encontrado.")
        return self._prompts[key]