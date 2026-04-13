# KB Case Project

Asistente conversacional para soporte tecnico y construccion de base de conocimiento. El sistema recibe mensajes en lenguaje natural, decide si el usuario esta haciendo una pregunta o reportando un incidente, y ejecuta un flujo controlado para responder, pedir aclaraciones, revisar conocimiento y persistir aprendizaje reutilizable.

## Objetivo

Convertir conversaciones operativas en conocimiento estructurado y util, sin perder trazabilidad ni control del proceso. El proyecto esta pensado para escenarios donde no basta con responder texto: tambien hay que clasificar, validar completitud, recuperar contexto, involucrar a un humano cuando hace falta y publicar conocimiento de forma segura.

## Que hace

- Acepta entrada tipo chat con `texto_usuario`.
- Distingue entre pregunta general e incidente tecnico.
- Convierte incidentes libres en una estructura formal (`EntradaIncidente`).
- Extrae un caso reutilizable (`CasoExtraido`) a partir del incidente.
- Evalua si hay informacion suficiente para guardar conocimiento.
- Solicita aclaraciones cuando faltan datos.
- Busca casos similares en un almacen vectorial.
- Decide si el caso es existente, variante o nuevo.
- Ejecuta revision humana con `aprobar`, `rechazar` o `editar`.
- Persiste el caso aprobado en almacenamiento duradero.
- Indexa el caso para recuperacion semantica futura.
- Devuelve siempre una respuesta textual al usuario.

## Arquitectura

El flujo esta implementado como un grafo de estados con nodos explicitos, transiciones condicionales y checkpoints por hilo de conversacion. Esto permite que cada etapa tenga una responsabilidad clara y que el sistema pueda pausar y reanudar la ejecucion sin perder el contexto operativo.

Resumen del flujo:

1. Interpretacion de entrada.
2. Enrutamiento a pregunta general o incidente.
3. Extraccion y evaluacion del incidente.
4. Ciclo de aclaraciones si la informacion es insuficiente.
5. Recuperacion de casos similares.
6. Decision sobre el conocimiento generado.
7. Revision humana.
8. Persistencia e indexacion.
9. Respuesta final.

## Capacidades avanzadas que le dan valor

### 1. Estado explicito y tipado

El sistema no depende de variables dispersas ni de payloads improvisados. Cada ejecucion evoluciona sobre un estado definido con campos claros como `incidente`, `caso_extraido`, `completitud`, `casos_similares`, `decision_caso`, `motivo_detencion` y `respuesta`.

Valor:
- Facilita depuracion y evolucion del flujo.
- Reduce ambiguedad entre etapas.
- Hace posible razonar sobre el sistema como software, no solo como automatizacion.

### 2. Reanudacion real por hilo

Cada conversacion puede pausarse y continuar con un `thread_id`. Esto permite pedir aclaraciones o revision humana y luego retomar exactamente desde el punto correcto.

Valor:
- Mantiene continuidad conversacional sin rehacer el flujo.
- Evita reconstruir contexto manualmente.
- Mejora robustez en procesos largos o asincronos.

### 3. Human-in-the-loop estructurado

La participacion humana no es un parche externo. El flujo puede interrumpirse, solicitar datos concretos o pedir una decision formal sobre el conocimiento propuesto.

Valor:
- Permite control humano en puntos sensibles.
- Hace posible aprobar, rechazar o editar antes de publicar.
- Aumenta calidad del conocimiento final.

### 4. Separacion entre conversacion y modelo operativo

El usuario interactua con texto libre, pero internamente el sistema trabaja con estructuras como `EntradaIncidente`, `CasoExtraido` y `DecisionCaso`.

Valor:
- La experiencia es natural para el usuario.
- El backend conserva rigor y datos explotables.
- Permite auditar y reutilizar informacion con precision.

### 5. Construccion incremental de conocimiento

El proyecto no solo responde; tambien transforma incidentes en activos reutilizables, los guarda y los indexa para futuras recuperaciones.

Valor:
- Cada interaccion puede enriquecer la base de conocimiento.
- El sistema mejora con el uso.
- La conversacion deja un resultado operativo durable.

### 6. Control fino del flujo

El comportamiento del sistema no depende de una sola llamada al modelo. Hay nodos, rutas condicionales, puntos de pausa y decisiones separadas por etapa.

Valor:
- Mayor previsibilidad.
- Menor acoplamiento entre pasos.
- Mejor capacidad para introducir reglas, politicas y validaciones.

## Por que estas cualidades importan

En soluciones centradas unicamente en automatizacion visual, es posible imitar parte de la experiencia externa, pero cuesta mucho mas conservar estas propiedades al mismo tiempo:

- continuidad exacta de la ejecucion
- estado intermedio claro y consistente
- pausas y reanudacion nativas
- validacion por etapas
- revision humana integrada en el runtime
- evolucion del sistema sin volverlo fragil

La diferencia practica no esta solo en lo que el usuario ve, sino en la calidad interna del sistema:

- mejor mantenibilidad
- mejor auditabilidad
- mejor capacidad de crecimiento
- menor fragilidad cuando el flujo se vuelve complejo

## Estructura principal

- [principal.py](/home/harold/Escritorio/Aprender/ag-langchain/src/04-practicas/nivel-03/kb-case-project/principal.py): sesion conversacional, `thread_id`, checkpoints y reanudacion.
- [constructor_grafo.py](/home/harold/Escritorio/Aprender/ag-langchain/src/04-practicas/nivel-03/kb-case-project/constructor_grafo.py): definicion del grafo y sus rutas.
- [nodos.py](/home/harold/Escritorio/Aprender/ag-langchain/src/04-practicas/nivel-03/kb-case-project/nodos.py): logica de interpretacion, extraccion, revision, persistencia y respuesta.
- [estado.py](/home/harold/Escritorio/Aprender/ag-langchain/src/04-practicas/nivel-03/kb-case-project/estado.py): estado compartido y contexto de ejecucion.
- [esquemas.py](/home/harold/Escritorio/Aprender/ag-langchain/src/04-practicas/nivel-03/kb-case-project/esquemas.py): contratos estructurados del dominio.
- [test.py](/home/harold/Escritorio/Aprender/ag-langchain/src/04-practicas/nivel-03/kb-case-project/test.py): rondas de prueba para recorrer las capacidades del flujo.

## Ejecucion

Ejecutar una consulta puntual:

```bash
python principal.py "Payments-api devuelve 500 en prod con timeout de base de datos"
```

Ejecutar rondas de prueba:

```bash
python test.py
```

## Estado actual del proyecto

El proyecto ya cubre:

- interfaz conversacional
- clasificacion de entrada
- extraccion estructurada
- aclaraciones iterativas
- revision humana
- persistencia de conocimiento
- indexacion vectorial
- respuesta final al usuario

El siguiente salto natural seria incorporar politicas de aprobacion, evaluacion automatica del flujo y memoria de largo plazo por tenant o por usuario.
