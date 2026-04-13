from __future__ import annotations

from typing import Any, Callable

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.postgres import PostgresStore
from langgraph.types import Command

from configuracion import URI_BD_STORE
from constructor_grafo import construir_grafo
from estado import ContextoAplicacion

InterruptResolver = Callable[[dict[str, Any]], dict[str, Any]]


def _obtener_interrupcion(grafo: Any, config: dict[str, Any]) -> dict[str, Any] | None:
    estado = grafo.get_state(config=config)
    if not getattr(estado, "next", None):
        return None

    tareas = getattr(estado, "tasks", [])
    if not tareas:
        return None

    interrupciones = getattr(tareas[0], "interrupts", [])
    if not interrupciones:
        return None

    return interrupciones[0].value


def resolver_interrupcion_por_defecto(interrupcion: dict[str, Any]) -> dict[str, Any]:
    tipo = interrupcion.get("tipo")

    if tipo == "solicitud_aclaracion":
        respuestas: list[dict[str, str]] = []
        for pregunta in interrupcion.get("preguntas", []):
            respuestas.append(
                {
                    "pregunta": str(pregunta),
                    "respuesta": (
                        "Confirmado: sucede en prod desde ayer, al crear pagos; "
                        "afecta a multiples usuarios y no hubo cambios de codigo "
                        "aplicados por nosotros."
                    ),
                }
            )
        return {"respuestas": respuestas}

    if tipo == "revision_conocimiento":
        return {"accion": "aprobar"}

    return {}


class ChatKBSession:
    def __init__(
            self,
            thread_id: str = "chat-kb-001",
            resolver_interrupcion: InterruptResolver | None = None,
            contexto: ContextoAplicacion | None = None,
    ) -> None:
        self.thread_id = thread_id
        self.resolver_interrupcion = resolver_interrupcion or resolver_interrupcion_por_defecto
        self.contexto = contexto or ContextoAplicacion(
            id_tenant="acme",
            espacio_nombres_oficial=("acme", "casos_oficiales"),
            espacio_nombres_candidato=("acme", "casos_candidatos"),
            maximo_rondas_aclaracion=3,
        )
        self.config = {"configurable": {"thread_id": thread_id}}
        self._store_cm: PostgresStore | None = None
        self.grafo: Any = None

    def __enter__(self) -> "ChatKBSession":
        builder = construir_grafo()
        checkpointer = InMemorySaver()
        self._store_cm = PostgresStore.from_conn_string(URI_BD_STORE)
        store = self._store_cm.__enter__()
        store.setup()
        self.grafo = builder.compile(store=store, checkpointer=checkpointer)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._store_cm is not None:
            self._store_cm.__exit__(exc_type, exc_val, exc_tb)
            self._store_cm = None
        self.grafo = None

    def run(
            self,
            query: str,
            resolver_interrupcion: InterruptResolver | None = None,
    ) -> dict[str, Any]:
        if self.grafo is None:
            raise RuntimeError("La sesion no esta abierta. Usa 'with ChatKBSession(...) as sesion'.")

        texto = query.strip()
        if not texto:
            return {"respuesta": "No se recibio ningun mensaje."}

        resolver = resolver_interrupcion or self.resolver_interrupcion

        resultado = self.grafo.invoke(
            {"texto_usuario": texto},
            context=self.contexto,
            config=self.config,
        )

        while True:
            interrupcion = _obtener_interrupcion(self.grafo, self.config)
            if interrupcion is None:
                break

            payload = resolver(interrupcion)
            resultado = self.grafo.invoke(
                Command(resume=payload),
                context=self.contexto,
                config=self.config,
            )

        return resultado


def ejecutar_query(
        query: str,
        thread_id: str = "chat-kb-001",
        resolver_interrupcion: InterruptResolver | None = None,
) -> dict[str, Any]:
    with ChatKBSession(
            thread_id=thread_id,
            resolver_interrupcion=resolver_interrupcion,
    ) as sesion:
        return sesion.run(query)


def main(query: str) -> None:
    resultado = ejecutar_query(query=query)
    print(resultado.get("respuesta", resultado))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit('Uso: python principal.py "<query>"')
    main(query=" ".join(sys.argv[1:]))
