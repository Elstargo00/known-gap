from dataclasses import dataclass


@dataclass(frozen=True)
class AskCommand:
    query: str
    mode: str = "normal"
