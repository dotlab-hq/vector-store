"""Authentication middleware.

Checks ``Authorization: Bearer <AUTH_SECRET>`` on every request except the
public pages (``/``, ``/playground``, ``/util``, ``/health``).

When ``AUTH_SECRET`` env var is empty, auth is disabled (dev convenience).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.config import settings

# Paths that never require auth.
_PUBLIC_PATHS: set[str] = {"/", "/playground", "/util", "/health"}


class AuthMiddleware(BaseHTTPMiddleware):
    """Validate Bearer token on every request.

    If ``AUTH_SECRET`` is not set, all requests pass through (dev mode).
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        # Auth is mandatory. When AUTH_SECRET is not configured, refuse
        # all protected requests so the operator notices.
        if not settings.auth_secret:
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "Server misconfigured: AUTH_SECRET not set. Set it in .env or environment variables."
                },
            )

        # Public paths are always open.
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        # Extract and validate Bearer token.
        auth_header = request.headers.get("authorization", "")
        scheme, _, token = auth_header.partition(" ")

        if scheme.lower() != "bearer" or not token:
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Missing or invalid Authorization header. Expected: Bearer <AUTH_SECRET>"
                },
            )

        if not _constant_time_compare(token, settings.auth_secret):
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid credentials"},
            )

        return await call_next(request)


def _constant_time_compare(a: str, b: str) -> bool:
    """Compare two strings in constant time to prevent timing attacks."""
    import hmac

    return hmac.compare_digest(a.encode(), b.encode())


__all__ = ["AuthMiddleware"]
