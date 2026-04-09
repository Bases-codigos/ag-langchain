from langchain.tools import BaseTool

from app.tools.calculator import calculator
from app.tools.kb_search import kb_search
from app.tools.profile_tool import build_profile_tool

class ToolRegistry:
    def get_tools(self, tool_names: list[str], user_facts: list[str]) -> list[BaseTool]:
        available = {
            "calculator": calculator,
            "kb_search": kb_search,
            "profile_tool": build_profile_tool(user_facts),
        }

        tools: list[BaseTool] = []
        for name in tool_names:
            tool = available.get(name)
            if tool is not None:
                tools.append(tool)

        return tools