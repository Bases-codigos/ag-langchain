# Memoria Jerarquica Persistente

## Que es

Es organizar la memoria en capas, segun el tipo de informacion que guardas y el tiempo durante el cual sigue siendo util.

No toda memoria es igual. Un buen sistema separa:

- memoria episodica
- memoria semantica
- preferencias

## 1. Memoria episodica

### Que es

Guarda lo que ocurrio en una conversacion o caso concreto.

### Ejemplos

- el usuario reporto error 500
- se pidieron aclaraciones
- el humano rechazo la publicacion

### Para que sirve

- continuar el mismo hilo
- reconstruir el historial de un caso
- entender que ya se hizo

### Ejemplo

```python
store.put(
    ("episodic", "thread-001"),
    "evt-001",
    {
        "type": "clarification_requested",
        "question": "Ocurrio en prod o staging?",
    },
)
```

## 2. Memoria semantica

### Que es

Hechos relativamente estables y reutilizables.

### Ejemplos

- `payments-api` usa PostgreSQL
- el tenant `acme` opera en `prod` y `staging`
- incidentes previos de `webhooks-api` suelen relacionarse con rotacion de claves

### Para que sirve

- enriquecer decisiones futuras
- recuperar patrones utiles
- dar contexto durable al sistema

### Ejemplo

```python
store.put(
    ("semantic", "tenant-facts"),
    "acme",
    {
        "critical_services": ["payments-api", "auth-api"],
        "database_engine": "postgresql",
    },
)
```

## 3. Preferencias

### Que es

Datos sobre como quiere interactuar el usuario o el tenant.

### Ejemplos

- idioma preferido
- estilo de respuesta
- nivel de detalle

### Para que sirve

- personalizar respuestas
- adaptar tono y formato

### Ejemplo

```python
store.put(
    ("preferences",),
    "u_123",
    {
        "language": "es",
        "style": "directo",
    },
)
```

## Por que jerarquica

Porque mezclar todo en un solo lugar degrada el sistema:

- cuesta saber que es estable y que es temporal
- la memoria se vuelve ruidosa
- recuperar contexto correcto es mas dificil

## Ejemplo simple de uso conjunto

```python
def build_prompt(runtime, thread_id: str, user_id: str) -> str:
    episodic = runtime.store.get(("episodic", thread_id), "latest")
    semantic = runtime.store.get(("semantic", "tenant-facts"), runtime.context.id_tenant)
    prefs = runtime.store.get(("preferences",), user_id)

    return f"""
Contexto episodico: {episodic.value if episodic else {}}
Hechos semanticos: {semantic.value if semantic else {}}
Preferencias: {prefs.value if prefs else {}}
"""
```

## Persistente significa

Que la memoria no se pierde al terminar la ejecucion actual. Vive fuera del proceso, por ejemplo:

- Postgres
- Redis
- un store persistente de LangGraph

## Como llevarlo a tu proyecto

Podrias tener:

- episodica: historial del caso actual, aclaraciones, aprobaciones
- semantica: hechos por servicio, patrones de fallo, resoluciones frecuentes
- preferencias: idioma, tono, nivel tecnico del usuario

## Regla practica

Si algo describe un caso puntual, es episodico.  
Si algo es un hecho reutilizable, es semantico.  
Si algo describe como responder, son preferencias.
