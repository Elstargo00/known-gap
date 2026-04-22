from dataclasses import dataclass


@dataclass(frozen=True)
class ExampleCommand:
    name: str
