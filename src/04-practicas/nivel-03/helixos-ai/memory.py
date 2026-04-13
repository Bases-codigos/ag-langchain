from langgraph.store.memory import InMemoryStore

# En desarrollo
store = InMemoryStore()

# Podrías pre cargar preferencias o facts de usuario
store.put(
    ("user_profile",),
    "u_123",
    {
        "communication_style": "direct",
        "preferred_language": "es",
        "known_services": ["billing", "webhooks", "api"]
    },
)

store.put(
    ("tenant_policies",),
    "acme_corp",
    {
        "refund_window_days": 7,
        "requires_human_for_prod_actions": True,
    }
)

# Store es la memoria persistente cross-thread. La organización recomendada
# namespace + key + json document, y para producción la documentación recomienda
# un store persistente como Postgres o Redis