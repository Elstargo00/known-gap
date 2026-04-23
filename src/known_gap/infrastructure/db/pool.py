import asyncpg
from pgvector.asyncpg import register_vector


async def create_pool(dsn: str, min_size: int = 1, max_size: int = 10) -> asyncpg.Pool:
    async def _init(conn: asyncpg.Connection) -> None:
        await register_vector(conn)

    pool = await asyncpg.create_pool(
        dsn=dsn,
        min_size=min_size,
        max_size=max_size,
        init=_init,
    )
    if pool is None:
        raise RuntimeError("Failed to create asyncpg pool")
    return pool
