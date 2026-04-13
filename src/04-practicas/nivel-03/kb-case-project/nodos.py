import uuid
from typing import Literal
from langgraph.runtime import Runtime
from langgraph.types import interrupt
from pydantic import BaseModel
from configuracion import modelo_llm, almacen_vectorial
from esquemas import (
    EntradaIncidente,
    CasoExtraido,
    EvaluacionInformacionFaltante,
    CasoSimilar,
    DecisionCaso,
    CasoConocimiento,
    )
from estado import EstadoKB, ContextoAplicacion
from utilidades import (
    obtener_fecha_utl_actual,
    convertir_caso_a_documento,
    normalizar_puntaje,
    construir_texto_incidente
    )

# ============================================================
# Nodos del grafo
# ============================================================
class ClasificacionEntrada(BaseModel):
    tipo_entrada: Literal["pregunta_general", "incidente_tecnico"]


def interpretar_entrada_usuario(
        state: EstadoKB,
        runtime: Runtime[ContextoAplicacion],
) -> dict:
    """
    Interpreta texto libre y decide si es pregunta general o incidente técnico.
    Si el incidente ya llega estructurado, lo clasifica directo como incidente.
    """
    incidente = state.get("incidente")
    if incidente is not None:
        if hasattr(incidente, "model_dump"):
            incidente = incidente.model_dump()
        return {
            "tipo_entrada": "incidente_tecnico",
            "incidente": incidente,
        }

    texto_usuario = state.get("texto_usuario", "").strip()
    if not texto_usuario:
        return {"tipo_entrada": "pregunta_general"}

    clasificador = modelo_llm.with_structured_output(ClasificacionEntrada)
    prompt = f"""
Clasifica este mensaje en una sola categoría:
- pregunta_general: consulta conceptual o informativa
- incidente_tecnico: reporte de error, fallo o problema operativo

Mensaje del usuario:
{texto_usuario}
"""
    salida = clasificador.invoke(prompt)
    return {"tipo_entrada": salida.tipo_entrada}


def enrutador_tipo_entrada(state: EstadoKB) -> str:
    if state.get("tipo_entrada") == "incidente_tecnico":
        return "construir_incidente_desde_texto"
    return "responder_pregunta"


def construir_incidente_desde_texto(
        state: EstadoKB,
        runtime: Runtime[ContextoAplicacion],
) -> dict:
    """
    Convierte entrada libre en EntradaIncidente cuando no viene estructurada.
    """
    if state.get("incidente") is not None:
        return {}

    texto_usuario = state.get("texto_usuario", "").strip()
    extractor = modelo_llm.with_structured_output(EntradaIncidente)
    prompt = f"""
Convierte este texto en un incidente técnico estructurado.

Reglas:
- Si falta id_incidente, crea uno con prefijo CHAT-INC.
- Si faltan mensaje_error o extracto_logs, usa "no_provisto".
- Si no se menciona resolución, usa resolucion_aplicada=null y resuelto=false.
- No inventes detalles fuera del texto.

Texto:
{texto_usuario}
"""
    incidente = extractor.invoke(prompt)
    return {"incidente": incidente.model_dump()}


def extraer_caso(state: EstadoKB, runtime: Runtime[ContextoAplicacion]) -> dict:
    """
    Convierte un incidente en un caso estructurado reutilizable.
    """
    incidente = EntradaIncidente(**state["incidente"])
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

    return {"caso_extraido": caso_extraido.model_dump()}

def evaluar_completitud(state: EstadoKB, runtime: Runtime[ContextoAplicacion]) -> dict:
    """
    Evalúa si hay suficiente información para guardar conocimiento útil.
    """
    incidente = EntradaIncidente(**state["incidente"])
    caso_extraido = CasoExtraido(**state["caso_extraido"])
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
    return {"completitud": completitud.model_dump()}

def enrutador_aclaraciones(state: EstadoKB) -> str:
    """
    Decide si hay que pedir aclaraciones, continuar o detener.
    """
    completitud = state["completitud"]
    rondas = state.get("rondas_aclaracion", 0)
    if completitud.get("esta_completo", False):
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
        "preguntas": completitud.get("preguntas_para_usuario", []),
        "campos_faltantes": completitud.get("campos_faltantes", []),
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
        "rondas_aclaracion": rondas + 1,
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
        caso_extraido["texto_canonico"],
        k = 5,
        filter = {"id_tenant": runtime.context.id_tenant}
    )
    casos_similares: list[dict] = []
    for indice, doc in enumerate(resultados):
        casos_similares.append(CasoSimilar(
            id_caso=doc.metadata["id_caso"],
            titulo=doc.metadata.get("titulo", "Sin titulo"),
            estado=doc.metadata.get("estado", "candidato"),
            puntaje=normalizar_puntaje(indice),
        ).model_dump())

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
resuelto={incidente.get("resuelto")}
resolucion_aplicada={incidente.get("resolucion_aplicada")}

Borrador:
{CasoExtraido(**caso_extraido).model_dump_json(indent=2)}

Similares:
{similares}
"""
    decision_caso = clasificador.invoke(prompt)
    return {"decision_caso": decision_caso.model_dump()}

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

Incidente origen: {incidente.get("id_incidente")}
Servicio: {incidente.get("servicio")}
Entorno: {incidente.get("entorno")}

Clasificación propuesta: {decision_caso.get("decision")}
Motivo: {decision_caso.get("motivo")}
Caso objetivo: {decision_caso.get("id_caso_objetivo")}

Titulo: {caso_extraido.get("titulo")}
Síntomas: {", ".join(caso_extraido.get("sintomas", []))}
Causa probable: {caso_extraido.get("causa_raiz_probable")}
Resolución: {" | ".join(caso_extraido.get("resolucion", []))}
Palabras clave: {", ".join(caso_extraido.get("palabras_clave", []))}
Confianza: {caso_extraido.get("confianza")}

Casos similares:
{chr(10).join(f"- {c.get('id_caso')} | {c.get('titulo')} | {c.get('estado')} | puntaje={c.get('puntaje')}" for c in similares) or "- ninguno"}
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
        "caso_propuesto": caso_extraido,
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
        return {"caso_extraido": caso_editado.model_dump()}

    return {}

def enrutador_revision(state: EstadoKB) -> str:
    """
    Si el caso fue rechazado, responde al usuario. Si no, persiste.
    """
    if state.get("motivo_detencion"):
        return "responder_resultado_incidente"
    return "persistir_caso"

def  persistir_caso(
        state: EstadoKB,
        runtime: Runtime[ContextoAplicacion],
) -> dict:
    """
    Guarda el caso final en el store persistente.
    """
    incidente = EntradaIncidente(**state["incidente"])
    caso_extraido = CasoExtraido(**state["caso_extraido"])

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

    return {"caso_persistido": caso.model_dump()}

def indexar_caso(
        state: EstadoKB,
        runtime: Runtime[ContextoAplicacion],
) -> dict:
    """
    Indexa el caso persistido en el almacén vectorial.
    """
    caso = CasoConocimiento(**state["caso_persistido"])

    doc = convertir_caso_a_documento(caso)
    doc.metadata["id_tenant"] = runtime.context.id_tenant

    almacen_vectorial.add_documents([doc])

    return {}


def responder_resultado_incidente(
        state: EstadoKB,
        runtime: Runtime[ContextoAplicacion],
) -> dict:
    """
    Devuelve una respuesta textual final para el flujo de incidente.
    """
    incidente = state.get("incidente")
    completitud = state.get("completitud")
    decision = state.get("decision_caso")
    similares = state.get("casos_similares", [])
    motivo_detencion = state.get("motivo_detencion")

    prompt = f"""
Eres un agente de soporte técnico.
Responde de forma clara, breve y accionable en español.

Formato de salida:
1) Diagnóstico breve
2) Próximos 3 pasos
3) Información adicional útil a pedir (si aplica)

Contexto:
- incidente: {incidente if incidente else "no_disponible"}
- completitud: {completitud if completitud else "no_disponible"}
- decision_caso: {decision if decision else "no_disponible"}
- casos_similares: {similares}
- motivo_detencion: {motivo_detencion or "ninguno"}
"""
    respuesta = modelo_llm.invoke(prompt)
    return {"respuesta": respuesta.content}


def responder_pregunta(
        state: EstadoKB,
        runtime: Runtime[ContextoAplicacion],
) -> dict:
    """
    Ruta para preguntas generales tipo chat.
    """
    texto_usuario = state.get("texto_usuario", "").strip()
    if not texto_usuario:
        return {"respuesta": "No se recibió un mensaje para responder."}

    prompt = f"""
Responde de forma clara y breve en español.

Pregunta del usuario:
{texto_usuario}
"""
    respuesta = modelo_llm.invoke(prompt)
    return {"respuesta": respuesta.content}
