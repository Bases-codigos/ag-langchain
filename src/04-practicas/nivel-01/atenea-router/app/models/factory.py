from langchain_openai import ChatOpenAI
from app.config import settings

class ModelFactory:
    @staticmethod
    def get_model(model_name: str):
        if model_name == settings.reasoning_model:
            return ChatOpenAI(
                model = model_name,
                temperature=settings.temperature_technical,
            )
        return ChatOpenAI(
            model=model_name,
            temperature=settings.temperature_general,
        )