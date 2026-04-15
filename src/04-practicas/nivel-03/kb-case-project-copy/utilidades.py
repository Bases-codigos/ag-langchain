# 4

from datetime import datetime, timezone
from langchain_core.documents import Document

from esquemas import EntradaIncidente, CasoConocimiento

# Utilidades

def obtener_fecha_util_actual() -> str:
    """Devuelve la fecha/hora actual en UTC con formato ISO."""
    return datetime.now(timezone.utc).isoformat()

def convertir_caso_a_documento(caso: CasoConocimiento) -> Document:
    """Convierte un caso persistido a un Document para indexarlo."""
    return Document(
        page_content=caso.texto_canonico,
        metadata = {
            "id_caso": caso.id_caso,
            "estado": caso.estado,
            "servicio": caso.servicio,
            "entorno": caso.entorno,
            "titulo": caso.titulo,
        }
    )

def normalizar_puntaje(posicion: int) -> float:
    """Asigna un puntaje simple basado en la posición del resultado."""
    return max(0.0, 1.0 - (posicion * 0.1))

def construir_texto_incidente(
        incidente: EntradaIncidente,
        aclaraciones: list[dict[str, str]],
) -> str:
    """
    Construye un bloque de texto con el incidente y sus aclaraciones.
    """
    bloque_aclaraciones = "\n".join(
        f"- P: {a['pregunta']}\n R: {a['respuesta']}" for a in aclaraciones) if aclaraciones else "(sin aclaraciones adicionales)"
    return f"""
id_incidente: {incidente.id_incidente}
servicio: {incidente.servicio}
entorno: {incidente.entorno}
resumen: {incidente.resumen}
mensaje_error: {incidente.mensaje_error}
extracto_logs: {incidente.extracto_logs}
resolucion_aplicada: {incidente.resolucion_aplicada}
resuelto: {incidente.resuelto}

Aclaración del usuario:
{bloque_aclaraciones}
""".strip()