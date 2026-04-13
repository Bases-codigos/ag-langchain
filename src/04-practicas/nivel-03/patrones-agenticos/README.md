# Patrones Agenticos

Esta carpeta resume patrones que aumentan el valor tecnico de un sistema construido con `LangChain` y `LangGraph`. El objetivo es aprender a pasar de un flujo lineal a un sistema con estado, reglas, memoria y colaboracion entre agentes.

## Contenido

1. [01-motor-de-estado-formal.md](/home/harold/Escritorio/Aprender/ag-langchain/src/04-practicas/nivel-03/patrones-agenticos/01-motor-de-estado-formal.md)
2. [02-policy-engine-real.md](/home/harold/Escritorio/Aprender/ag-langchain/src/04-practicas/nivel-03/patrones-agenticos/02-policy-engine-real.md)
3. [03-memoria-jerarquica-persistente.md](/home/harold/Escritorio/Aprender/ag-langchain/src/04-practicas/nivel-03/patrones-agenticos/03-memoria-jerarquica-persistente.md)
4. [04-multiagente-serio.md](/home/harold/Escritorio/Aprender/ag-langchain/src/04-practicas/nivel-03/patrones-agenticos/04-multiagente-serio.md)
5. [05-planner-replanner-critic.md](/home/harold/Escritorio/Aprender/ag-langchain/src/04-practicas/nivel-03/patrones-agenticos/05-planner-replanner-critic.md)

## Orden recomendado

1. Empieza por el motor de estado formal.
2. Sigue con policy engine.
3. Luego memoria jerarquica.
4. Termina con multiagente serio.
5. Profundiza con planner/replanner/critic (version didactica y version real).

## Idea central

Estos patrones se combinan bien:

- el motor de estado controla el ciclo de vida
- el policy engine decide que esta permitido
- la memoria guarda lo importante entre turnos y sesiones
- el sistema multiagente reparte el trabajo entre especialistas

Cuando los unes, dejas de tener solo un flujo con LLM y pasas a tener un runtime agentico mas serio.
