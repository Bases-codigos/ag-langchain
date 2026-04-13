# Definimos el estado del grafo y el contexto

from dataclasses import dataclass
from typing import Any
from typing import Literal
from typing_extensions import TypedDict, NotRequired

# ============================================================
# Estado del grafo
# ============================================================
class EstadoKB(TypedDict, total=False):
    """
    Estado global de flujo

    No todos los campos existen desde el inicio; muchos aparecen
    cuando alún nodo los produce
    """
    texto_usuario: str
    tipo_entrada: Literal["pregunta_general", "incidente_tecnico"]
    incidente: dict[str, Any]

    caso_extraido: NotRequired[dict[str, Any]]
    completitud: NotRequired[dict[str, Any]]
    rondas_aclaracion: NotRequired[int]
    aclaraciones_usuario: NotRequired[list[dict[str, str]]]
    casos_similares: NotRequired[list[dict[str, Any]]]
    decision_caso: NotRequired[dict[str, Any]]
    resumen_revision: NotRequired[str]
    caso_persistido: NotRequired[dict[str, Any]]
    motivo_detencion: NotRequired[str]
    respuesta: NotRequired[str]

# ============================================================
# Contexto del runtime
# ============================================================
# El contexto NO es el state.
# Es información externa y estable para la ejecución, por ejemplo:
# - tenant actual
# - namespaces
# - límites de configuración
# ============================================================
@dataclass
class ContextoAplicacion:
    id_tenant: str
    espacio_nombres_oficial: tuple[str, str]
    espacio_nombres_candidato: tuple[str, str]
    maximo_rondas_aclaracion: int = 3
