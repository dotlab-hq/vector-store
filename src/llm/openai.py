import os

from langchain_openai import ChatOpenAI

from src.config import settings

base_url = (
    settings.openai_base_url
    or os.environ.get("OPENAI_BASE_URL")
    or os.environ.get("OPENAI_API_BASE")
)

llm = ChatOpenAI(
    model=settings.openai_chat_model,
    temperature=settings.generation_temperature,
    api_key=settings.openai_api_key,
    base_url=base_url,
)
