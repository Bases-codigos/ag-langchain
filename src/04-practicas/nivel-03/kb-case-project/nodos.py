import uuid
from langgraph.runtime import Runtime
from langgraph.types import interrupt
from configuracion import modelo_llm, almacen_vectorial
from esquemas import (
    CasoExtraido,
    EvaluacionInformacionFaltante,
    CasoSimilar,
    DecisionCaso,
    CasoConocimiento,
    )
from estado import EstadoKB, ContextoAplicacion, EstadoChat
from utilidades import (
    obtener_fecha_utl_actual,
    convertir_caso_a_documento,
    normalizar_puntaje,
    construir_texto_incidente
    )

# ============================================================
# Nodos del grafo
# ============================================================
def extraer_caso(state: EstadoKB, runtime: Runtime[ContextoAplicacion]) -> dict:
    """
    Convierte un incidente en un caso estructurado reutilizable.
    """
    incidente = state["incidente"]
    aclaraciones = state.get("aclaraciones_usuario", [])

    extractor = modelo_llm.with_structured_output(CasoExtraido)

    prompt = f"""
Eres un analista senior de incidentes.
Convierte este incidente en un borrador de caso reutilizable.

Reglas:
- No inventes datos.
- Si la información aún no es clara, dela la confianza más baja.
- texto_canonico debe servir para embeddings.

{construir_texto_incidente(incidente, aclaraciones)}
"""
    caso_extraido = extractor.invoke(prompt)

    return {"caso_extraido": caso_extraido}

def evaluar_completitud(state: EstadoKB, runtime: Runtime[ContextoAplicacion]) -> dict:
    """
    Evalúa si hay suficiente información para guardar conocimiento útil.
    """
    incidente = state["incidente"]
    caso_extraido = state["caso_extraido"]
    aclaraciones = state.get("aclaraciones_usuario", [])

    evaluador = modelo_llm.with_structured_output(EvaluacionInformacionFaltante)

    prompt = f"""
Evalúa si hay información suficiente para guardar conocimiento útil.

Consider incompleto su falta claridad suficiente para guardar conocimiento útil.

Considera incompleto si falta claridad práctica sobre:
- cómo se resolvió
- bajo qué condiciones ocurrió
- qué evidencia apoya la causa o la solución

Devuelve preguntas concretas y cortas solo si realmente hacen falta.

Incidente y aclaraciones:
{construir_texto_incidente(incidente, aclaraciones)}

Borrador del caso:
{caso_extraido.model_dump_json(indent=2)}
"""
    completitud = evaluador.invoke(prompt)
    return {"completitud": completitud}

def enrutador_aclaraciones(state: EstadoKB) -> str:
    """
    Decide si hay que pedir aclaraciones, continuar o detener.
    """
    completitud = state["completitud"]
    rondas = state.get("rondas_aclaracion", 0)
    if completitud.esta_completo:
        return "buscar_casos_similares"

    if rondas >= 3:
        return "detener_caso_incompleto"

    return "pedir_aclaracion_usuario"

def pedir_aclaracion_usuario(
        state: EstadoKB,
        runtime: Runtime[ContextoAplicacion],
) -> dict:
    """
    Interrumpe el flujo para pedir más información al usuario.
    """
    completitud = state["completitud"]
    rondas = state.get("rondas_aclaracion", 0)

    carga = {
        "tipo": "solicitud_aclaracion",
        "ronda": rondas + 1,
        "preguntas": completitud.preguntas_para_usuario,
        "campos_faltantes": completitud.campos_faltantes,
        "instruccion": "Responde las preguntas con el mayor detalle posible."
    }

    respuesta_humana = interrupt(carga)

    # Se espera algo como:
    # {
    #   "respuestas": [
    #       {"pregunta": "...", "respuesta": "..."}
    #   ]
    # }
    respuestas = respuesta_humana.get("respuestas", [])
    existentes = state.get("aclaraciones_usuario", [])

    return {
        "aclaraciones_usuario": existentes + respuestas,
        "rondas_aclaraciones": rondas + 1,
    }

def detener_caso_incompleto(
        state: EstadoKB,
        runtime: Runtime[ContextoAplicacion],
) -> dict:
    """
    Finaliza cuando no se pudo completar el caso tras varias rondas.
    """
    return {
        "motivo_detencion": (
            "No se pudo completar el caso con suficiente claridad "
            "después del número máximo de rondas de aclaración."
        )
    }

def buscar_casos_similares(
        state: EstadoKB,
        runtime: Runtime[ContextoAplicacion],
) -> dict:
    """
    Busca casos semánticamente parecidos en el vector store.
    """
    caso_extraido = state["caso_extraido"]

    resultados = almacen_vectorial.similarity_search(
        caso_extraido.texto_canonico,
        k = 5,
        filter = {"id_tenant": runtime.context.id_tenant}
    )
    casos_similares: list[CasoSimilar] = []
    for indice, doc in enumerate(resultados):
        casos_similares.append(CasoSimilar(
            id_caso=doc.metadata["id_caso"],
            titulo=doc.metadata.get("titulo", "Sin titulo"),
            estado=doc.metadata.get("estado", "candidato"),
            puntaje=normalizar_puntaje(indice),
        ))

    return {"casos_similares": casos_similares}

def decidir_accion_caso(
        state: EstadoKB,
        runtime: Runtime[ContextoAplicacion],
) -> dict:
    """
    Decide si el caso corresponde a uno existente, una variante o uno nuevo.
    """
    incidente = state["incidente"]
    caso_extraido = state["caso_extraido"]
    similares = state.get("casos_similares", [])

    clasificador = modelo_llm.with_structured_output(DecisionCaso)

    prompt = f"""
Clasifica el borrador en una de estas categorías:
- existente: ya es esencialmente el mismo caso
- variante: variante útil de un caso conocido
- nuevo: caso nuevo

Si es existente y hay un caso objetivo claro, devuelve id_caso_objetivo.

incidente:
resuelto={incidente.resuelto}
resolucion_aplicada={incidente.resolucion_aplicada}

Borrador:
{caso_extraido.model_dump_json(indent=2)}

Similares:
{[x.model_dump() for x in similares]}
"""
    decision_caso = clasificador.invoke(prompt)
    return {"decision_caso": decision_caso}

def preparar_revision_humana(
        state: EstadoKB,
        runtime: Runtime[ContextoAplicacion],
) -> dict:
    """
    Prepara un resumen claro para un revisor humano.
    """
    incidente = state["incidente"]
    caso_extraido = state["caso_extraido"]
    decision_caso = state["decision_caso"]
    similares = state.get("casos_similares", [])

    resumen = f"""
Se propone añadir o actualizar conocimientos especializado.

Incidente origen: {incidente.id_incidente}
Servicio: {incidente.servicio}
Entorno: {incidente.entorno}

Clasificación propuesta: {decision_caso.decision}
Motivo: {decision_caso.motivo}
Caso objetivo: {decision_caso.id_caso_objetivo}

Titulo: {caso_extraido.titulo}
Síntomas: {", ".join(caso_extraido.sintomas)}
Causa probable: {caso_extraido.causa_raiz_probable}
Resolución: {" | ".join(caso_extraido.resolucion)}
Palabras clave: {", ".join(caso_extraido.palabras_clave)}
Confianza: {caso_extraido.confianza}

Casos similares:
{chr(10).join(f"- {c.id_caso} | {c.titulo} | {c.estado} | puntaje={c.puntaje}" for c in similares) or "- ninguno"}
"""
    return {"resumen_revision": resumen}

def compuerta_revision_humana(
        state: EstadoKB,
        runtime: Runtime[ContextoAplicacion],
) -> dict:
    """
    Segundo punto de human-in-the-loop.
    """
    caso_extraido = state["caso_extraido"]
    resumen = state["resumen_revision"]

    carga = {
        "tipo": "revision_conocimiento",
        "resumen": resumen,
        "caso_propuesto": caso_extraido.model_dump(),
        "acciones_permitidas": ["aprobar", "rechazar", "editar"],
        "instrucciones": (
            "Aprueba, rechaza o edita el caso antes de publicarlo."
            "Si editas, devuelve el caso completo corregido."
        )
    }

    revision = interrupt(carga)
    accion = revision.get("accion")
    if accion == "rechazar":
        return {"motivo_detencion": "El humano rechazó la publicación del conocimiento"}

    if accion == "editar":
        caso_editado = CasoExtraido(**revision["caso_editado"])
        return {"caso_extraido": caso_editado}

    return {}

def enrutador_revision(state: EstadoKB) -> str:
    """
    Si el caso fue rechazado, termina. Si no, persiste
    """
    if state.get("motivo_detencion"):
        return "END"
    return "persistir_caso"

def  persistir_caso(
        state: EstadoKB,
        runtime: Runtime[ContextoAplicacion],
) -> dict:
    """
    Guarda el caso final en el store persistente.
    """
    incidente = state["incidente"]
    caso_extraido = state["caso_extraido"]

    ahora = obtener_fecha_utl_actual()

    caso = CasoConocimiento(
        id_caso=str(uuid.uuid4()),
        estado="oficial",
        titulo=caso_extraido.titulo,
        servicio=caso_extraido.servicio,
        entorno=caso_extraido.entorno,
        sintomas=caso_extraido.sintomas,
        causa_raiz_probable=caso_extraido.causa_raiz_probable,
        resolucion=caso_extraido.resolucion,
        palabras_clave=caso_extraido.palabras_clave,
        confianza=caso_extraido.confianza,
        incidentes_origen=[incidente.id_incidente],
        texto_canonico=caso_extraido.texto_canonico,
        creado_en=ahora,
        actualizado_en=ahora,
    )

    runtime.store.put(
        runtime.context.espacio_nombres_candidato,
        caso.id_caso,
        caso.model_dump(),
    )

    return {"caso_persistido": caso}

def indexar_caso(
        state: EstadoKB,
        runtime: Runtime[ContextoAplicacion],
) -> dict:
    """
    Indexa el caso persistido en el almacén vectorial.
    """
    caso = state["caso_persistido"]

    doc = convertir_caso_a_documento(caso)
    doc.metadata["id_tenant"] = runtime.context.id_tenant

    almacen_vectorial.add_documents([doc])

    return {}

def responder_pregunta(
        state: EstadoChat,
        runtime: Runtime[ContextoAplicacion],
) -> dict:
    pregunta = state["pregunta"]
    prompt = f"""
Responde de forma clara y breve a la siguiente pregunta:

Pregunta:
{pregunta}
"""
    respuesta = modelo_llm.invoke(prompt)
    return {"respuesta": respuesta.contenido}