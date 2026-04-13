from langgraph.graph import StateGraph, START, END

from estado import EstadoKB, ContextoAplicacion
from nodos import (
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
    responder_pregunta,
)

# ============================================================
# Construcción del grafo
# ============================================================
def construir_grafo():
    builder = StateGraph(EstadoKB, context_schema=ContextoAplicacion)

    # Registro de nodos
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

    # Flujo principal
    builder.add_edge(START, "extraer_caso")
    builder.add_edge("extraer_caso", "evaluar_completitud")

    builder.add_conditional_edges(
        "evaluar_completitud",
        enrutador_aclaraciones,
        {
            "pedir_aclaracion_usuario": "pedir_aclaracion_usuario",
            "buscar_casos_similares": "buscar_casos_similares",
            "detener_caso_incompleto": "detener_caso_incompleto",
        }
    )

    builder.add_edge("pedir_aclaracion_usuario", "extraer_caso")
    builder.add_edge("detener_caso_incompleto", END)

    builder.add_edge("buscar_casos_similares", "decidir_accion_caso")
    builder.add_edge("decidir_accion_caso", "preparar_revision_humana")
    builder.add_edge("preparar_revision_humana", "compuerta_revision_humana")

    builder.add_conditional_edges(
        "compuerta_revision_humana",
        enrutador_revision,
        {
            "persistir_caso": "persistir_caso",
            "END": END
        }
    )

    builder.add_edge("persistir_caso", "indexar_caso")
    builder.add_edge("indexar_caso", END)

    return builder