"""Mint a JWT signed with JWT_SECRET from .env for manual API testing.

Local development helper only. The secret is read from settings and never
printed or logged; only the signed token and its subject/expiry are emitted.

Usage:
    uv run python scripts/mint_jwt.py
    uv run python scripts/mint_jwt.py --user-id 11111111-1111-1111-1111-111111111111
    uv run python scripts/mint_jwt.py --expires-hours 1
"""

import argparse
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import jwt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.known_gap.config.settings import get_settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mint a local-dev JWT for /ask and /ingest.",
    )
    parser.add_argument(
        "--user-id",
        type=uuid.UUID,
        default=None,
        help="Subject UUID. Defaults to a fresh UUID4 (fresh graph per token).",
    )
    parser.add_argument(
        "--expires-hours",
        type=int,
        default=24,
        help="Token lifetime in hours (default: 24).",
    )
    args = parser.parse_args()

    settings = get_settings()
    if not settings.jwt_secret:
        parser.error("JWT_SECRET is not configured in .env")

    user_id: uuid.UUID = args.user_id or uuid.uuid4()
    issued_at = datetime.now(UTC)
    expires_at = issued_at + timedelta(hours=args.expires_hours)

    token = jwt.encode(
        {"sub": str(user_id), "iat": issued_at, "exp": expires_at},
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )

    print(f"User ID:  {user_id}")
    print(f"Expires:  {expires_at.isoformat()}")
    print()
    print(f"Bearer {token}")


if __name__ == "__main__":
    main()
