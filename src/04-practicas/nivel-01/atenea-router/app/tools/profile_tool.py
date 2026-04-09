from langchain.tools import tool

def build_profile_tool(user_facts: list[str]):
    @tool
    def user_profile(_: str = "") -> str:
        """
        Devuelve hechos persistentes del usuario
        """
        if not user_facts:
            return "No hay información persistente del usuario."
        return "Hechos del usuario:\n-" + "\n- ".join(user_facts)
    
    return user_profile