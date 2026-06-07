from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "vector-store"
    debug: bool = False

    # Auth — plain-text bearer token required on all API routes.
    # When empty, auth is disabled (dev convenience).  Set via AUTH_SECRET env var.
    auth_secret: str = ""

    # OpenAI
    openai_api_key: str = ""
    openai_base_url: str = "http://localhost:25789/v1"
    openai_chat_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    # PostgreSQL
    database_url: str = ""

    # Redis
    redis_url: str = ""

    # LangSmith
    langsmith_api_key: str = ""
    langsmith_project: str = "vector-store"
    langsmith_tracing: bool = False

    # Embedding
    embedding_dimension: int = 1536

    # Chunking
    parent_chunk_max_tokens: int = 1024
    child_chunk_max_tokens: int = 256

    # Retrieval
    retrieval_top_k: int = 20
    rerank_top_k: int = 10
    fusion_k: int = 60  # RRF constant

    # Generation
    generation_temperature: float = 0.3
    max_context_tokens: int = 8000

    # S3 / MinIO Storage
    s3_bucket: str = "vector-store"
    s3_region: str = "us-east-1"
    s3_endpoint: str = ""  # MinIO endpoint (leave empty for AWS S3)
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_presign_expiry: int = 900  # presigned URL lifetime in seconds (15 min)

    # Neo4j Knowledge Graph
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"
    neo4j_enabled: bool = False  # disabled by default until configured

    # Qdrant Vector Store
    qdrant_url: str = ""  # e.g. http://localhost:6333 — leave empty for in-memory
    qdrant_api_key: str = ""  # required for Qdrant Cloud
    qdrant_prefer_grpc: bool = False
    qdrant_path: str = ""  # local on-disk path — leave empty for in-memory
    qdrant_collection: str = "rag_chunks"

    # Vector Stores worker / cron
    vector_store_worker_concurrency: int = 10
    vector_store_worker_poll_interval_s: float = 2.0
    vector_store_worker_lease_minutes: int = 10
    vector_store_retry_base_s: int = 30
    vector_store_retry_cap_s: int = 3600
    vector_store_retry_max: int = 5
    vector_store_cron_interval_s: float = 10.0
    vector_store_search_fanout: int = 4
    vector_store_worker_enabled: bool = True


settings = Settings()
