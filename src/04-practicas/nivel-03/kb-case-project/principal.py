from langgraph.store.postgres import PostgresStore

from configuracion import URI_BD_STORE
from esquemas import EntradaIncidente
from estado import ContextoAplicacion
from constructor_grafo import construir_grafo

# ============================================================
# Punto de entrada del programa
# ============================================================
def main():
    builder = construir_grafo()

    with PostgresStore.from_conn_string(URI_BD_STORE) as store:
        store.setup()

        grafo = builder.compile(store=store)
        resultado = grafo.invoke(
            {
                "incidente": EntradaIncidente(
                    id_incidente="INC-2026-00421",
                    servicio="payments-api",
                    entorno="prod",
                    resumen="Errores intermitentes al crear pagos",
                    mensaje_error="psycopg.OperationalError: connection timeout expired",
                    extracto_logs="pool checkout failed; timeout after 30s",
                    resolucion_aplicada=None,
                    resuelto=False,
                )
            },
            context=ContextoAplicacion(
                id_tenant="acme",
                espacio_nombres_oficial=("acme", "casos_oficiales"),
                espacio_nombres_candidato=("acme", "casos_candidatos"),
                maximo_rondas_aclaracion=3,
            ),
        )
        print("Resultado final del flujo:")
        print(resultado)

if __name__ == "__main__":
    main()