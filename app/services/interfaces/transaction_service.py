from abc import ABC, abstractmethod
from typing import Optional
from app.models.transaction_dto import CreateTransactionRequest, TransactionResponse


class ITransactionService(ABC):
    @abstractmethod
    def create_transaction(
        self, data: CreateTransactionRequest
    ) -> Optional[TransactionResponse]:
        pass
