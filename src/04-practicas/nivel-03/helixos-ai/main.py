from context import AppContext
from graph import build_graph
from langchain.messages import HumanMessage

def main() -> None:
    graph = build_graph()

    context = AppContext(
        user_id="u_123",
        tenant_id="acme_corp",
        plan="enterprise",
        can_issue_refunds=True,
        environment="prod"
    )

    config = {
        "configurable": {
            "thread_id": "thread-support-001"
        }
    }

    result = graph.invoke({
        "user_message": (
            "Mi factura fue cobrada dos veces y además el webhook de producción "
            "está devolviendo errores 500 desde anoche."
        ),
        "notes": [],
        "actions_taken": [],
        "escalations": [],
    },
        config = config,
        context = context,
    )

    print("\n=== RESPUESTA FINAL ===\n")
    print(result["final_response"])
    print("\n=== ACCIONES ===\n")
    print(result["actions_taken"])

if __name__ == "__main__":
    main()