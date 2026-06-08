import os

from langchain_openai import OpenAIEmbeddings

from src.config import settings

base_url = (
    settings.openai_base_url
    or os.environ.get("OPENAI_BASE_URL")
    or os.environ.get("OPENAI_API_BASE")
    or None
)

embeddings = OpenAIEmbeddings(
    model=settings.openai_embedding_model,
    dimensions=settings.embedding_dimension,
    api_key=settings.openai_api_key or None,
    base_url=base_url,
    check_embedding_ctx_length=False,
)
