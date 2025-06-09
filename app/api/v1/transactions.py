from fastapi import APIRouter, Depends, HTTPException
from app.models.transaction_dto import CreateTransactionRequest, TransactionResponse
from app.services.implementation.transaction_service import TransactionService
from app.services.interfaces.transaction_service import ITransactionService
from app.infrastructure.implementation.transaction_repository import (
    TransactionRepository,
)

router = APIRouter()

# Dependency Injection
transaction_service: ITransactionService = TransactionService(TransactionRepository())


@router.post("/", response_model=TransactionResponse)
def create_transaction(data: CreateTransactionRequest):
    created = transaction_service.create_transaction(data)
    if not created:
        raise HTTPException(status_code=400, detail="User creation failed")
    return TransactionResponse(**created.__dict__)
