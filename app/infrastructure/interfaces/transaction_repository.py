from abc import ABC, abstractmethod
from typing import List, Optional
from app.domain.transaction import Transaction


class ITransactionRepository(ABC):
    @abstractmethod
    def create_transaction(
        self, transaction: Transaction
    ) -> Optional[Transaction]:
        pass

    @abstractmethod
    def get_user_transactions(self, user_id: str) -> List[Transaction]:
        pass
