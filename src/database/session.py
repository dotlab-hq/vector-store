from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.config import settings


def _build_database_url(url: str) -> str:
    """Ensure the URL uses the asyncpg driver and strip unsupported params."""
    # Switch to the async driver if not already set
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)

    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    params.pop("channel_binding", None)
    params.pop("sslmode", None)
    clean_query = urlencode(params, doseq=True)
    clean_url = urlunparse(parsed._replace(query=clean_query))
    return clean_url


def _connect_args(url: str) -> dict:
    """Extract SSL config for asyncpg from the database URL."""
    params = parse_qs(urlparse(url).query)
    sslmode = params.get("sslmode", [None])[0]
    if sslmode in ("require", "verify-full", "verify-ca", "prefer"):
        import ssl as ssl_module

        ctx = ssl_module.create_default_context()
        if sslmode == "require":
            ctx.check_hostname = False
            ctx.verify_mode = ssl_module.CERT_NONE
        return {"ssl": ctx}
    return {}


_raw_url = settings.database_url
_clean_url = _build_database_url(_raw_url)
_connect = _connect_args(_raw_url)

engine = create_async_engine(
    _clean_url,
    echo=settings.debug,
    pool_pre_ping=True,
    connect_args=_connect,
)

async_session_factory = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)
