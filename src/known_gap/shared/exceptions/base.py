from dataclasses import dataclass, field
from typing import Any


@dataclass
class AppException(Exception):
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    http_status_code: int = 500
    error_code: str = "INTERNAL_ERROR"
    is_retryable: bool = False

    def __str__(self) -> str:
        return self.message


@dataclass
class TransientException(AppException):
    http_status_code: int = 503
    is_retryable: bool = True


@dataclass
class PermanentException(AppException):
    http_status_code: int = 400
    is_retryable: bool = False
