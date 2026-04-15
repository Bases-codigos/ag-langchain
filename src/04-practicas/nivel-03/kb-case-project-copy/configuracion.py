# 1
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGEngine, PGVectorStore
from dotenv import load_dotenv
from sqlalchemy.exc import ProgrammingError

load_dotenv()

# Configuración general

# Base de datos para el Store de LangGraph
URI_BD_STORE = "postgresql://postgres:postgres@172.16.21.60:5656/agent_memory?sslmode=disable"

# Base de datos para PGVectorStore
URI_DB_VECTOR = "postgresql+psycopg://postgres:postgres@172.16.21.60:56567agent_memory"

# Modelo principal
modelo_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# Modelo de embeddings
modelo_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Motor de conexión a PostgreSQL para vectores
motor_vectorial = PGEngine.from_connection_string(uri=URI_BD_VECTOR)

# Crear tabla vectorial si aún no existe
# text-embedding-3-small normalmente usa 1536 dimensiones
try:
    motor_vectorial.init_vectorstore_table(
        table_name="casos_errores_empresa",
        vector_size=1536,
    )
except ProgrammingError as exc:
    # Si la tabla ya existe, continuamos sin fallar.
    if "already exists" not in str(exc).lower():
        raise

# Vector store moderno
almacen_vectorial = PGVectorStore.create_sync(
    engine=motor_vectorial,
    table_name="casos_errores_empresa",
    embedding_service=modelo_embeddings,
)

