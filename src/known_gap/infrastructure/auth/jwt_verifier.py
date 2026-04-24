from uuid import UUID

import jwt

from src.known_gap.shared.exceptions.base import PermanentException


class JWTVerifier:
    """Verifies JWTs minted by the frontend and returns the user_id.

    Expects HS256 signatures with a shared secret and claims
    {sub: <user-id UUID>, iat, exp}. Signature, expiration, and iat are
    all verified; iss and aud are not enforced in Phase 4.
    """

    def __init__(self, secret: str, algorithm: str = "HS256") -> None:
        self._secret = secret
        self._algorithm = algorithm

    def verify(self, token: str) -> UUID:
        try:
            payload = jwt.decode(
                token,
                self._secret,
                algorithms=[self._algorithm],
                options={"require": ["exp", "iat", "sub"]},
            )
        except jwt.ExpiredSignatureError as e:
            raise PermanentException(
                message="Token has expired",
                error_code="TOKEN_EXPIRED",
                http_status_code=401,
            ) from e
        except jwt.InvalidTokenError as e:
            raise PermanentException(
                message=f"Invalid token: {e}",
                error_code="INVALID_TOKEN",
                http_status_code=401,
            ) from e

        sub = payload.get("sub")
        if not isinstance(sub, str):
            raise PermanentException(
                message="Token missing valid 'sub' claim",
                error_code="INVALID_TOKEN",
                http_status_code=401,
            )
        try:
            return UUID(sub)
        except ValueError as e:
            raise PermanentException(
                message="Token 'sub' is not a valid UUID",
                error_code="INVALID_TOKEN",
                http_status_code=401,
            ) from e
