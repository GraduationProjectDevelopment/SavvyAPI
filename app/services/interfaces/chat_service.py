from abc import ABC, abstractmethod


class IChatService(ABC):
    @abstractmethod
    def get_response(self, user_id: str, message: str) -> str:
        pass
