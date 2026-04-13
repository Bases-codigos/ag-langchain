from dataclasses import dataclass

# frozen=True significa que la instancia es inmutable después de creada
@dataclass
class Settings(frozen=True):
    default_model: str = "gpt-4o-mini"
    reasoning_model: str = "gpt-4.1"
    temperature_general: float = 0.3
    temperature_technical: float = 0.1
    max_short_term_messages: int = 8

settings = Settings()