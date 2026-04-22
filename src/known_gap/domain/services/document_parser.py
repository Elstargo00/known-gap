from abc import ABC, abstractmethod


class DocumentParser(ABC):
    @abstractmethod
    def parse(self, filename: str, data: bytes) -> str: ...
