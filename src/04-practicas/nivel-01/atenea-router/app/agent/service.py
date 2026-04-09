from langchain.agents import create_agent
from app.memory.long_term import LongTermMemory
from app.memory.short_term import ShortTermMemory
from app.models.factory import ModelFactory
from app.prompts.manager import PromptManager
from app.routing.agent_router import AgentRouter
from app.schemas import AgentResponse, UserMessage
from app.tools.registry import ToolRegistry

class AgentService:
    """
    Servicio principal del agente orquestador.

    Responsabilidades:
    - decidir configuración dinámica mediante routing
    - recuperar memoria corta y larga
    - construir agent en tiempo de ejecución
    - ejecutar la interacción
    - persistir contexto relevante
    """

    def __init__(self,
                 router: AgentRouter,
                 model_factory: ModelFactory,
                 prompt_manager: PromptManager,
                 tool_registry: ToolRegistry,
                 short_memory: ShortTermMemory,
                 long_memory: LongTermMemory,) -> None:
        self.router = router
        self.model_factory = model_factory
        self.prompt_manager = prompt_manager
        self.tool_registry = tool_registry
        self.short_memory = short_memory
        self.long_memory = long_memory

    def _extract_memory_fact(self, user_message: str) -> str:
        """
        Extrae un hecho persistente desde frases del tipo:
        - 'recuerda que ...'
        - 'a partir de ahora ...'
        - 'no olvides que ...'
        """
        prefixes = ["recuerda que", "a partir de ahora", "no olvides que"]
        lowered = user_message.lower()

        for prefix in prefixes:
            if prefix in lowered:
                start_idx = lowered.find(prefix) + len(prefix)
                return user_message[start_idx:].strip(" .")

        return user_message.strip()

    def _build_memory_context(self, user_id: str, thread_id: str) -> str:
        """
        Construye un bloque textual con memoria larga y memoria corta.
        Este bloque se inyecta dentro del system prompt dinámico.
        """
        facts = self.long_memory.get_facts(user_id)
        history = self.short_memory.get(thread_id)

        long_term_block = (
            "Memoria larga del usuario:\n- " + "\n- ".join(facts)
            if facts
            else "Memoria larga del usuario:\n- Sin datos persistentes."
        )

        if history:
            short_term_lines = [
                f"- {item['role']}: {item['content']}"
                for item in history
            ]
            short_term_block = "Memoria corta reciente:\n" + "\n".join(short_term_lines)
        else:
            short_term_block = "Memoria corta reciente:\n- Sin historial reciente."

        return f"{long_term_block}\n\n{short_term_block}"

    def _build_system_prompt(self, base_prompt: str, memory_context: str) -> str:
        """
        Une la política base del agente con el contexto de memoria.
        """
        return (
            f"{base_prompt}\n\n"
            f"Contexto adicional disponible para personalizar la respuesta:\n"
            f"{memory_context}"
        )

    def invoke(self, payload: UserMessage) -> AgentResponse:
        """
        Ejecuta una interacción completa del agente.
        """
        decision = self.router.route(payload.message)

        if decision.mode == "memory_update":
            fact = self._extract_memory_fact(payload.message)
            self.long_memory.add_fact(payload.user_id, fact)

            self.short_memory.append(payload.thread_id, "human", payload.message)
            confirmation = f"He guardado esta preferencia: {fact}"
            self.short_memory.append(payload.thread_id, "ai", confirmation)

            return AgentResponse(
                answer=confirmation,
                used_model=decision.model_name,
                used_tools=[],
                active_prompt=decision.system_prompt_key,
            )

        user_facts = self.long_memory.get_facts(payload.user_id)
        tools = self.tool_registry.get_tools(decision.tool_names, user_facts)
        model = self.model_factory.get_model(decision.model_name)

        base_prompt = self.prompt_manager.get(decision.system_prompt_key)
        memory_context = self._build_memory_context(
            user_id=payload.user_id,
            thread_id=payload.thread_id,
        )
        system_prompt = self._build_system_prompt(base_prompt, memory_context)

        agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
        )

        history = self.short_memory.get(payload.thread_id)
        messages: list[dict[str, str]] = []

        for item in history:
            messages.append(
                {
                    "role": "user" if item["role"] == "human" else "assistant",
                    "content": item["content"],
                }
            )

        messages.append({"role": "user", "content": payload.message})

        result = agent.invoke({"messages": messages})

        answer = self._extract_final_answer(result)

        self.short_memory.append(payload.thread_id, "human", payload.message)
        self.short_memory.append(payload.thread_id, "ai", answer)

        return AgentResponse(
            answer=answer,
            used_model=decision.model_name,
            used_tools=decision.tool_names,
            active_prompt=decision.system_prompt_key,
        )

    def _extract_final_answer(self, result: dict) -> str:
        """
        Normaliza la salida del agente.

        Dependiendo del runtime/configuración, LangChain puede devolver
        una estructura con 'messages'. Aquí tomamos el último mensaje útil.
        """
        messages = result.get("messages", [])

        if not messages:
            return "No se obtuvo respuesta del agente."

        last_message = messages[-1]

        content = getattr(last_message, "content", None)
        if isinstance(content, str):
            return content

        if isinstance(last_message, dict):
            raw_content = last_message.get("content")
            if isinstance(raw_content, str):
                return raw_content

        return str(last_message)