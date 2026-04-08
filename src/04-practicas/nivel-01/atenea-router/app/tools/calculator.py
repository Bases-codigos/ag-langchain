from langchain.tools import tool

@tool
def calculator(expression: str) -> str:
    """
    Evalúa expresiones matemáticas básicas
    :param expression:
    :return: str
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Resultado: {result}"
    except Exception as e:
        return f"Error al calcular: {e}"