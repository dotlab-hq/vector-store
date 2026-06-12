FROM python:3.14-slim AS builder
WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
RUN uv sync --no-cache --no-install-project --extra api

FROM python:3.14-slim AS production
WORKDIR /app

RUN pip install --no-cache-dir uv

COPY --from=builder /app/.venv ./.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY src/config/       ./src/config/
COPY src/database/     ./src/database/
COPY src/generation/   ./src/generation/
COPY src/graph/        ./src/graph/
COPY src/indexing/     ./src/indexing/
COPY src/llm/          ./src/llm/
COPY src/observability/ ./src/observability/
COPY src/retrieval/    ./src/retrieval/
COPY src/shared/       ./src/shared/
COPY src/storage/      ./src/storage/
COPY src/vector_stores/ ./src/vector_stores/
COPY apps/api/         ./apps/api/
COPY entrypoint.sh     /entrypoint.sh

RUN chmod +x /entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
CMD ["api"]
