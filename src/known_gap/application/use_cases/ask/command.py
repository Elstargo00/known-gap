from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class AskCommand:
    user_id: UUID
    query: str
    mode: str = "normal"
