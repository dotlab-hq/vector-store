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

COPY src/              ./src/
COPY apps/api/         ./apps/api/
COPY entrypoint.sh     /entrypoint.sh

RUN chmod +x /entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
CMD ["api"]
