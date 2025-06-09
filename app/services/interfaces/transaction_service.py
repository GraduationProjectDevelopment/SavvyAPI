from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID
from app.models.transaction_dto import CreateTransactionRequest, TransactionResponse
from app.domain.transaction import Transaction


class ITransactionService(ABC):
    @abstractmethod
    def create_transaction(
        self, data: CreateTransactionRequest
    ) -> Optional[TransactionResponse]:
        pass

    @abstractmethod
    def get_transaction_by_id(
        self, transaction_id: UUID
    ) -> Optional[TransactionResponse]:
        pass

    @abstractmethod
    def get_transactions_by_user_id(self, user_id: UUID) -> List[TransactionResponse]:
        pass

    @abstractmethod
    def get_all_transactions(self) -> List[TransactionResponse]:
        pass

    @abstractmethod
    def update_transaction(
        self, transaction: Transaction
    ) -> Optional[TransactionResponse]:
        pass

    @abstractmethod
    def delete_transaction(self, transaction_id: UUID) -> bool:
        pass
    
    @abstractmethod
    def get_transactions_by_category_id(self, category_id: UUID) -> List[TransactionResponse]:
        pass
