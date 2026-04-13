# Definimos el estado del grafo y el contexto

from dataclasses import dataclass
from typing_extensions import TypedDict, NotRequired

from esquemas import (
    EntradaIncidente,
    CasoExtraido,
    EvaluacionInformacionFaltante,
    CasoSimilar,
    DecisionCaso,
    CasoConocimiento,
)

# ============================================================
# Estado del grafo
# ============================================================
class EstadoKB(TypedDict):
    """
    Estado global de flujo

    No todos los campos existen desde el inicio; muchos aparecen
    cuando alún nodo los produce
    """
    incidente: EntradaIncidente

    caso_extraido: NotRequired[CasoExtraido]
    completitud: NotRequired[EvaluacionInformacionFaltante]
    rondas_aclaracion: NotRequired[int]
    aclaraciones_usuario: NotRequired[list[dict[str, str]]]
    casos_similares: NotRequired[list[CasoSimilar]]
    decision_caso: NotRequired[DecisionCaso]
    resumen_revision: NotRequired[str]
    caso_persistido: NotRequired[CasoConocimiento]
    motivo_detencion: NotRequired[str]

# ============================================================
# Contexto del runtime
# ============================================================
# El contexto NO es el state.
# Es información externa y estable para la ejecución, por ejemplo:
# - tenant actual
# - namespaces
# - límites de configuración
# ============================================================
class EstadoChat(TypedDict):
    pregunta: str
    respuesta: NotRequired[str]


@dataclass
class ContextoAplicacion:
    id_tenant: str
    espacio_nombres_oficial: tuple[str, str]
    espacio_nombres_candidato: tuple[str, str]
    maximo_rondas_aclaracion: int = 3