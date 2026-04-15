# 2
from typing import Literal, Optional, Any
from pydantic import BaseModel, Field

# Esquemas / modelos de datos
class EntradaIncidente(BaseModel):
    """Incidente de entrada que llega al sistema"""
    id_incidente: str
    servicio: str
    entorno: str
    resumen: str
    mensaje_error: str
    extracto_logs: str
    resolucion_aplicada: Optional[str] = None
    resuelto: bool = False

class CasoExtraido(BaseModel):
    """Caso estructurado generado por el modelo a partir del incidente"""
    titulo: str
    servicio: str
    entorno: str
    sintomas: list[str]
    causa_raiz_probable: str
    resolucion: list[str]
    palabras_clave: list[str]
    confianza: float = Field(ge=0.0, le=1.0)
    texto_canonico: str

class EvaluacionInformacionFaltante(BaseModel):
    """
    Indica si el caso ya esá suficientemente completo o si
    todavía hacen falta aclaraciones.
    """
    esta_completo: bool
    campos_faltantes: list[str]
    preguntas_para_usuario: list[str]

class CasoSimilar(BaseModel):
    """Representa un caso parecido recuperado desde el vector store."""
    id_caso: str
    titulo: str
    puntaje: float
    estado: Literal["oficial", "candidato"]

class DecisionCaso(BaseModel):
    """
    Clasificación del caso respecto al conocimiento existente.
    """
    decision: Literal["existente", "variante", "nuevo"]
    motivo: str
    id_caso_objetivo: Optional[str] = None

class CasoConocimiento(BaseModel):
    """
    Caso final persistido en la base de conocimiento.
    """
    id_caso: str
    estado: Literal["oficial", "candidato"]
    titulo: str
    servicio: str
    entorno: str
    sintomas: list[str]
    causa_raiz_probable: str
    resolucion: list[str]
    palabras_clave: list[str]
    confianza: float
    incidente_origen: list[str]
    texto_canonico: str
    creado_en: str
    actualizado_en: str

class CargaRevision(BaseModel):
    """Modelo opcional para documentar cómo sería una revisión humana."""
    resumen: str
    caso_propuesto: dict[str, Any]
    decision: Literal["aprobar", "rechazar", "editar"]
    caso_editado: Optional[dict[str, Any]] = None