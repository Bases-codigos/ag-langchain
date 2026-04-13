from langgraph.graph import StateGraph, START, END

from estado import EstadoKB, ContextoAplicacion
from nodos import (
    interpretar_entrada_usuario,
    enrutador_tipo_entrada,
    construir_incidente_desde_texto,
    responder_pregunta,
    extraer_caso,
    evaluar_completitud,
    enrutador_aclaraciones,
    pedir_aclaracion_usuario,
    detener_caso_incompleto,
    buscar_casos_similares,
    decidir_accion_caso,
    preparar_revision_humana,
    compuerta_revision_humana,
    enrutador_revision,
    persistir_caso,
    indexar_caso,
    responder_resultado_incidente,
)


def construir_grafo():
    builder = StateGraph(EstadoKB, context_schema=ContextoAplicacion)

    builder.add_node("interpretar_entrada_usuario", interpretar_entrada_usuario)
    builder.add_node("construir_incidente_desde_texto", construir_incidente_desde_texto)
    builder.add_node("responder_pregunta", responder_pregunta)
    builder.add_node("extraer_caso", extraer_caso)
    builder.add_node("evaluar_completitud", evaluar_completitud)
    builder.add_node("pedir_aclaracion_usuario", pedir_aclaracion_usuario)
    builder.add_node("detener_caso_incompleto", detener_caso_incompleto)
    builder.add_node("buscar_casos_similares", buscar_casos_similares)
    builder.add_node("decidir_accion_caso", decidir_accion_caso)
    builder.add_node("preparar_revision_humana", preparar_revision_humana)
    builder.add_node("compuerta_revision_humana", compuerta_revision_humana)
    builder.add_node("persistir_caso", persistir_caso)
    builder.add_node("indexar_caso", indexar_caso)
    builder.add_node("responder_resultado_incidente", responder_resultado_incidente)

    # Entrada híbrida: texto humano o incidente ya estructurado.
    builder.add_edge(START, "interpretar_entrada_usuario")
    builder.add_conditional_edges(
        "interpretar_entrada_usuario",
        enrutador_tipo_entrada,
        {
            "responder_pregunta": "responder_pregunta",
            "construir_incidente_desde_texto": "construir_incidente_desde_texto",
        },
    )

    # Pregunta general.
    builder.add_edge("responder_pregunta", END)

    # Flujo técnico completo de incidentes.
    builder.add_edge("construir_incidente_desde_texto", "extraer_caso")
    builder.add_edge("extraer_caso", "evaluar_completitud")

    builder.add_conditional_edges(
        "evaluar_completitud",
        enrutador_aclaraciones,
        {
            "pedir_aclaracion_usuario": "pedir_aclaracion_usuario",
            "buscar_casos_similares": "buscar_casos_similares",
            "detener_caso_incompleto": "detener_caso_incompleto",
        },
    )

    builder.add_edge("pedir_aclaracion_usuario", "extraer_caso")
    builder.add_edge("detener_caso_incompleto", "responder_resultado_incidente")

    builder.add_edge("buscar_casos_similares", "decidir_accion_caso")
    builder.add_edge("decidir_accion_caso", "preparar_revision_humana")
    builder.add_edge("preparar_revision_humana", "compuerta_revision_humana")

    builder.add_conditional_edges(
        "compuerta_revision_humana",
        enrutador_revision,
        {
            "persistir_caso": "persistir_caso",
            "responder_resultado_incidente": "responder_resultado_incidente",
        },
    )

    builder.add_edge("persistir_caso", "indexar_caso")
    builder.add_edge("indexar_caso", "responder_resultado_incidente")
    builder.add_edge("responder_resultado_incidente", END)

    return builder
