# 3

from dataclasses import dataclass
from typing import Literal, Optional, Any
from typing_extensions import TypedDict, NotRequired

# Estado del grafo
class EstadoKB(TypedDict, total=False):
    """
    Estado global de flujo

    No todos los campos existen desde el inicio; muchos aparecen
    cuando algún nodo los produce
    """
    texto_usuario: str
    tipo_entrada: Literal["pregunta_general", "incidente_tecnico"]
    incidente: dict[str, Any]

    caso_extraido: NotRequired[dict[str, Any]]
    completitud: NotRequired[dict[str, Any]]
    rondas_aclaraciones: NotRequired[int]
    aclaraciones_usuario: NotRequired[dict[str, Any]]
    casos_similares: NotRequired[list[dict[str, Any]]]
    decision_caso: NotRequired[dict[str, Any]]
    resumen_revision: NotRequired[str]
    caso_persistido: NotRequired[dict[str, Any]]
    motivo_detencion: NotRequired[str]
    respuesta: NotRequired[str]

# Contexto del runtime
@dataclass
class ContextAplicacion:
    id_tenant: str
    espacio_nombre_oficial: tuple[str, str]
    espacio_nombre_candidato: tuple[str, str]
    maximo_rondas_aclaracion: int = 3









