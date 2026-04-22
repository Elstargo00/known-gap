from dataclasses import dataclass


@dataclass(frozen=True)
class ExampleResult:
    message: str
    success: bool = True
