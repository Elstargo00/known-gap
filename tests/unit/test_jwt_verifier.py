from datetime import UTC, datetime, timedelta
from uuid import uuid4

import jwt
import pytest

from src.known_gap.infrastructure.auth.jwt_verifier import JWTVerifier
from src.known_gap.shared.exceptions.base import PermanentException

SECRET = "a" * 32  # 32 bytes silences PyJWT's InsecureKeyLengthWarning for HS256


def _mint(
    *,
    sub: str | None,
    expires_in: timedelta = timedelta(minutes=5),
    secret: str = SECRET,
    extra: dict[str, object] | None = None,
) -> str:
    now = datetime.now(tz=UTC)
    payload: dict[str, object] = {"iat": now, "exp": now + expires_in}
    if sub is not None:
        payload["sub"] = sub
    if extra:
        payload.update(extra)
    return jwt.encode(payload, secret, algorithm="HS256")


class TestJWTVerifier:
    def test_verifies_valid_token_and_returns_sub_uuid(self) -> None:
        user_id = uuid4()
        token = _mint(sub=str(user_id))
        assert JWTVerifier(SECRET).verify(token) == user_id

    def test_rejects_expired_token(self) -> None:
        token = _mint(sub=str(uuid4()), expires_in=timedelta(seconds=-1))
        with pytest.raises(PermanentException) as excinfo:
            JWTVerifier(SECRET).verify(token)
        assert excinfo.value.error_code == "TOKEN_EXPIRED"
        assert excinfo.value.http_status_code == 401

    def test_rejects_token_signed_with_wrong_secret(self) -> None:
        token = _mint(sub=str(uuid4()), secret="b" * 32)
        with pytest.raises(PermanentException) as excinfo:
            JWTVerifier(SECRET).verify(token)
        assert excinfo.value.error_code == "INVALID_TOKEN"

    def test_rejects_token_missing_sub(self) -> None:
        token = _mint(sub=None)
        with pytest.raises(PermanentException) as excinfo:
            JWTVerifier(SECRET).verify(token)
        assert excinfo.value.error_code == "INVALID_TOKEN"

    def test_rejects_non_uuid_sub(self) -> None:
        token = _mint(sub="not-a-uuid")
        with pytest.raises(PermanentException) as excinfo:
            JWTVerifier(SECRET).verify(token)
        assert excinfo.value.error_code == "INVALID_TOKEN"
