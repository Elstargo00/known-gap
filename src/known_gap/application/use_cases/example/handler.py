from src.known_gap.application.use_cases.example.command import ExampleCommand
from src.known_gap.application.use_cases.example.result import ExampleResult


class ExampleHandler:
    async def execute(self, command: ExampleCommand) -> ExampleResult:
        return ExampleResult(message=f"Processed: {command.name}")
