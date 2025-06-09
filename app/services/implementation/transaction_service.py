from typing import Optional
from app.services.interfaces.transaction_service import ITransactionService
from app.models.transaction_dto import CreateTransactionRequest, TransactionResponse
from app.domain.transaction import Transaction, TransactionType
from app.infrastructure.interfaces.transaction_repository import ITransactionRepository
from app.core.Add_Trans import (
    process_transaction_with_llm,
    parse_llm_response,
    get_category_id,
)
from uuid import uuid4
from datetime import datetime


class TransactionService(ITransactionService):
    def __init__(self, transaction_repository: ITransactionRepository):
        self.transaction_repository = transaction_repository

    def create_transaction(
        self, data: CreateTransactionRequest
    ) -> Optional[TransactionResponse]:

        raw_response = process_transaction_with_llm(data.input_text)
        parsed = parse_llm_response(raw_response)

        category_id = get_category_id(parsed["category"])
        transaction = Transaction(
            transaction_id=uuid4(),
            user_id=data.user_id,
            category_id=category_id,
            description=parsed["description"],
            created_at=datetime.strptime(parsed["created_at"], "%Y-%m-%d %H:%M:%S+00"),
            amount=parsed["amount"],
            transaction_type=TransactionType(parsed["transaction_type"]),
        )
        response = self.transaction_repository.create_transaction(transaction)

        return TransactionResponse(
            transaction_id=response.transaction_id,
            user_id=response.user_id,
            category_id=response.category_id,
            description=response.description,
            created_at=response.created_at,
            amount=response.amount,
            transaction_type=response.transaction_type,
            feedback=parsed["feedback"],
        )
