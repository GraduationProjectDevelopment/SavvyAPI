from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID
from app.domain.transaction import Transaction


class ITransactionRepository(ABC):
    @abstractmethod
    def create_transaction(self, transaction: Transaction) -> Optional[Transaction]:
        pass

    @abstractmethod
    def get_transaction_by_id(self, transaction_id: UUID) -> Optional[Transaction]:
        pass

    @abstractmethod
    def get_transactions_by_user_id(self, user_id: UUID) -> List[Transaction]:
        pass

    @abstractmethod
    def get_all_transactions(self) -> List[Transaction]:
        pass

    @abstractmethod
    def update_transaction(self, transaction: Transaction) -> Optional[Transaction]:
        pass

    @abstractmethod
    def delete_transaction(self, transaction_id: UUID) -> bool:
        pass

    @abstractmethod
    def get_transactions_by_category_id(self, category_id: UUID) -> List[Transaction]:
        pass
    
