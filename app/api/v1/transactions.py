from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from app.domain.transaction import Transaction
from app.models.transaction_dto import (
    CreateTransactionRequest,
    TransactionResponse,
    UpdateTransactionRequest,
)
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


@router.get("/", response_model=List[TransactionResponse])
def get_all():
    return transaction_service.get_all_transactions()


@router.get("/{transaction_id}", response_model=TransactionResponse)
def get_by_id(transaction_id: UUID):
    tx = transaction_service.get_transaction_by_id(transaction_id)
    if not tx:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return tx


@router.get("/user/{user_id}", response_model=List[TransactionResponse])
def get_by_user(user_id: UUID):
    return transaction_service.get_transactions_by_user_id(user_id)


@router.put("/", response_model=TransactionResponse)
def update_transaction(data: UpdateTransactionRequest):
    tx = Transaction(**data.dict())
    updated = transaction_service.update_transaction(tx)
    if not updated:
        raise HTTPException(status_code=404, detail="Transaction not updated")
    return updated


@router.delete("/{transaction_id}", status_code=204)
def delete_transaction(transaction_id: UUID):
    success = transaction_service.delete_transaction(transaction_id)
    if not success:
        raise HTTPException(status_code=404, detail="Transaction not found")


@router.get("/category/{category_id}", response_model=List[TransactionResponse])
def get_by_category(category_id: UUID):
    return transaction_service.get_transactions_by_category_id(category_id)
