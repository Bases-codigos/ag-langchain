from __future__ import annotations

from typing import Any

from principal import ChatKBSession, resolver_interrupcion_por_defecto


def resolver_revision_rechazo(interrupcion: dict[str, Any]) -> dict[str, Any]:
    if interrupcion.get("tipo") == "revision_conocimiento":
        return {"accion": "rechazar"}
    return resolver_interrupcion_por_defecto(interrupcion)


def resolver_revision_edicion(interrupcion: dict[str, Any]) -> dict[str, Any]:
    if interrupcion.get("tipo") == "revision_conocimiento":
        caso = dict(interrupcion.get("caso_propuesto", {}))
        if caso:
            caso["titulo"] = f'{caso.get("titulo", "Caso")} [editado]'
            resolucion = caso.get("resolucion", [])
            if isinstance(resolucion, list):
                resolucion.append("Agregar alerta de saturacion de pool en monitoreo.")
                caso["resolucion"] = resolucion
        return {"accion": "editar", "caso_editado": caso}
    return resolver_interrupcion_por_defecto(interrupcion)


def main() -> None:
    rondas = [
        {
            "id": "R1",
            "objetivo": "Pregunta general (Q&A directo)",
            "query": (
                "Que es un connection timeout en PostgreSQL y cuales son "
                "3 mitigaciones practicas en produccion?"
            ),
        },
        {
            "id": "R2",
            "objetivo": "Incidente completo + persistencia + indexacion + revision aprobada",
            "query": (
                "Incidente en prod: payments-api devolvio 500 al crear pagos. "
                "Error: psycopg.OperationalError: connection timeout expired. "
                "Logs: pool checkout failed; timeout after 30s. "
                "Resolucion aplicada: aumentar temporalmente pool y reiniciar workers. "
                "Resuelto: si."
            ),
        },
        {
            "id": "R3",
            "objetivo": "Incidente incompleto para activar solicitud de aclaraciones",
            "query": (
                "Tenemos fallas intermitentes en payments-api en produccion. "
                "No tengo aun causa raiz ni resolucion confirmada."
            ),
        },
        {
            "id": "R4",
            "objetivo": "Revision humana con rechazo",
            "query": (
                "Incidente en prod: webhooks-api responde 500 al validar firmas. "
                "Error: invalid signature under load. "
                "Resolucion aplicada: rotacion de secret y ajuste de retries. "
                "Resuelto: si."
            ),
            "resolver": resolver_revision_rechazo,
        },
        {
            "id": "R5",
            "objetivo": "Revision humana con edicion",
            "query": (
                "Incidente en prod: auth-api con latencia extrema y timeouts de DB. "
                "Error: could not obtain connection from pool. "
                "Resolucion aplicada: limitar concurrencia y optimizar consulta lenta. "
                "Resuelto: si."
            ),
            "resolver": resolver_revision_edicion,
        },
        {
            "id": "R6",
            "objetivo": "Pregunta posterior en el mismo hilo",
            "query": (
                "Con base en incidentes de timeout de base de datos, "
                "que checklist operativo recomiendas para prevenir recurrencias?"
            ),
        },
    ]

    thread_id = "test-rondas-kb-001"
    print(f"Iniciando rondas en thread_id={thread_id}")

    with ChatKBSession(thread_id=thread_id) as sesion:
        for ronda in rondas:
            print("\n" + "=" * 80)
            print(f'{ronda["id"]} | {ronda["objetivo"]}')
            print("- Query:")
            print(ronda["query"])

            resultado = sesion.run(
                query=ronda["query"],
                resolver_interrupcion=ronda.get("resolver"),
            )
            respuesta = str(resultado.get("respuesta", resultado))
            respuesta_corta = respuesta if len(respuesta) <= 1000 else f"{respuesta[:1000]}..."

            print("- Respuesta:")
            print(respuesta_corta)

    print("\nRondas finalizadas.")


if __name__ == "__main__":
    main()
