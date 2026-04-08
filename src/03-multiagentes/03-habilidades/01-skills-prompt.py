from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# =========================================================
# 1) Repositorio simple de habilidades
#    En un caso real esto podría venir de archivos, BD, S3, etc.
# =========================================================
SKILLS = {
    "write_sql": """
    Eres un experto en SQL.
    
Tu trabajo es ayudar a escribir consultas SQL correctas, claras y seguras.

Reglas:
- Primero identifica el motor SQL si el usuario lo menciona (PostgreSQL, MySQL, SQLite, SQL Server, etc.).
- Si faltan nombres de tablas o columnas, indícalo claramente.
- Prefiere consultas legibles y bien formateadas.
- Evita DELETE o UPDATE sin WHERE, a menos que el usuario lo pida explícitamente.
- Si el usuario pide optimización, sugiere índices o mejoras de estructuras cuando tenga sentido.
- Si das una consulta, explica brevemente qué hace.

Formato de respuesta:
1. Explicación breve
2. Consulta SQL en bloque de código
3. Notas o supuestos, si aplica
""".strip(),
    "review_legal_doc": """
Eres un revisor de documentos legales.

Tu trabajo es analizar contratos, acuerdos o documentos legales y señalar:
- cláusulas ambiguas
- riesgos importantes
- obligaciones de cada parte
- fechas, plazos y penalidades
- términos que podrían requerir revisión profesional

Reglas:
- No afirmes que estás dando asesoría legal definitiva.
- Indica que se trata de una revisión orientativa.
- Resume en lenguaje claro.
- Si detectas una cláusula riesgosa, explica por qué.
- Si faltan partes del documento, dilo explícitamente.

Formato de respuesta:
1. Resumen general
2. Riesgos o puntos de atención
3. Cláusulas importantes
4. Recomendación final orientativa
    """.strip(),
}

# =========================================================
# 2) Herramienta para cargar habilidades bajo demanda
# =========================================================
@tool
def load_skill(skill_name: str) -> str:
    """Carga el prompt de una habilidad especializada.

Habilidades disponibles:
- write_sql: experto en redacción de consultas SQL
- review_legal_doc: revisor de documentos legales.

Devuelve el prompt y el contexto de la habilidad.
    """

    skill = SKILLS.get(skill_name)

    if not skill:
        available = ", ".join(SKILLS.keys())
        return (
            f"La habilidad '{skill_name}' no existe.\n"
            f"Habiliades disponibles: {available}"
        )
    return f"HABILIDAD CARGADA: {skill_name}\n\n{skill}"

# =========================================================
# 3) Modelo moderno
#    Cambia el modelo por el proveedor que uses realmente
# =========================================================
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# =========================================================
# 4) Agente principal
# =========================================================
agent = create_agent(
    model=model,
    tools=[load_skill],
    system_prompt=(
        "Eres un asistente útil. "
        "Tienes acceso a dos habilidades especializadas: "
        "'write_sql' y 'review_legal_doc'. "
        "Cuando el usuario pida ayuda con SQL, primero usa la herramienta "
        "load_skill con 'write_sql'. "
        "Cuando el usuario pida revisar contratos o documentos legales, "
        "primero usa la herramienta load_skill con 'review_legal_doc'. "
        "Después de cargar la habilidad, responde aplicando sus instrucciones. "
        "Si la petición no requiere una habilidad, responde normalmente."
    ),
)

# =========================================================
# 5) Ejemplos de uso
# =========================================================
if __name__ == "__main__":
    # Ejemplo 1: SQL
    result_sql = agent.invoke(HumanMessage(content=(
        "Necesito una consulta SQL para obtener los 10 clientes "
        "con más compras del último mes. "
        "La tabla de ventas se llama ventas y tiene: "
        "cliente_id, monto, fecha_compra."
    )))

    print("\n========== RESPUESTA SQL ==========\n")
    for msg in result_sql["messages"]:
        msg.pretty_print()

    # Ejemplo 2: revisión legal
    result_legal = agent.invoke(HumanMessage(content=(
        "Revisa esta cláusula del contrato: "
        "'El proveedor podrá modificar unilateralmente los plazos "
        "de entrega sin penalidad alguna.'"
    )))

    print("\n========== RESPUESTA LEGAL ==========\n")
    for msg in result_legal["messages"]:
        msg.pretty_print()