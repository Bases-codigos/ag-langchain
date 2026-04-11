# Contexto estático de ejecución
from dataclasses import dataclass
from typing import Literal


@dataclass
class AppContext:
    user_id: str
    tenant_id: str
    plan: str
    can_issue_refunds: bool
    environment: Literal["dev", "staging", "prod"]

# Runtime.context está pensado para inyectar dependencias o configuración estable
# por ejecución, como user id, permisos o conexiones. Eso evita meter datos estáticos
# dentro del state conversacional