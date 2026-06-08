# Multi-stage Dockerfile for Agentic RAG

# Stage 1: Builder
FROM python:3.14-slim AS builder

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies
RUN uv sync --no-cache --no-install-project

# Stage 2: Production
FROM python:3.14-slim AS production

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy virtual environment from builder
COPY --from=builder /app/.venv ./.venv

# Copy application code
COPY src/ ./src/
COPY apps/ ./apps/
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set PATH to use venv
ENV PATH="/app/.venv/bin:$PATH"

# Expose API port
EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["api"]