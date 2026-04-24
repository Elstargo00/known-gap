from falkordb.asyncio import FalkorDB

from src.known_gap.config.settings import Settings


def create_falkordb_client(settings: Settings) -> FalkorDB:
    """Build a FalkorDB async client from Settings.

    Connection is lazy — no network I/O happens until the first query.
    """
    return FalkorDB(
        host=settings.falkordb_host,
        port=settings.falkordb_port,
        username=settings.falkordb_username or None,
        password=settings.falkordb_password or None,
    )
