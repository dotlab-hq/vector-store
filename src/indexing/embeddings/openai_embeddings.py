from langchain_openai import OpenAIEmbeddings

from src.config import settings

embeddings = OpenAIEmbeddings(
    model=settings.openai_embedding_model,
    dimensions=settings.embedding_dimension,
    check_embedding_ctx_length=False,
)
