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

)