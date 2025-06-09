from typing import List, Optional
from app.infrastructure.interfaces.transaction_repository import ITransactionRepository
from app.domain.transaction import Transaction, TransactionType
from app.core.supabase import create_client, get_supabase
from uuid import UUID
import os




class TransactionRepository(ITransactionRepository):
    def create_transaction(self, transaction: Transaction) -> Optional[Transaction]:
        data = {
            "transaction_id": str(transaction.transaction_id),
            "user_id": str(transaction.user_id),
            "category_id": str(transaction.category_id),
            "description": transaction.description,
            "created_at": transaction.created_at.isoformat(),
            "amount": transaction.amount,
            "transaction_type": transaction.transaction_type.value,
        }
        response = get_supabase().table("transactions").insert(data).execute()
        if response.data:
            return transaction
        return None

    def get_transaction_by_id(self, transaction_id: UUID) -> Optional[Transaction]:
        response = (
            get_supabase().table("transactions")
            .select("*")
            .eq("transaction_id", str(transaction_id))
            .single()
            .execute()
        )
        if response.data:
            return Transaction(**response.data)
        return None

    def get_transactions_by_user_id(self, user_id: UUID) -> List[Transaction]:
        response = (
            get_supabase().table("transactions")
            .select("*")
            .eq("user_id", str(user_id))
            .execute()
        )
        return (
            [Transaction(**record) for record in response.data] if response.data else []
        )

    def get_all_transactions(self) -> List[Transaction]:
        response = get_supabase().table("transactions").select("*").execute()
        return (
            [Transaction(**record) for record in response.data] if response.data else []
        )

    def update_transaction(self, transaction: Transaction) -> Optional[Transaction]:
        response = (
            get_supabase().table("transactions")
            .update(
                {
                    "category_id": str(transaction.category_id),
                    "description": transaction.description,
                    "amount": transaction.amount,
                    "transaction_type": transaction.transaction_type.value,
                }
            )
            .eq("transaction_id", str(transaction.transaction_id))
            .execute()
        )
        return transaction if response.data else None

    def delete_transaction(self, transaction_id: UUID) -> bool:
        response = (
            get_supabase().table("transactions")
            .delete()
            .eq("transaction_id", str(transaction_id))
            .execute()
        )
        return bool(response.data)
    
    
    def get_transactions_by_category_id(self, category_id: UUID) -> List[Transaction]:
        response = (
            get_supabase().table("transactions")
            .select("*")
            .eq("category_id", str(category_id))
            .execute()
        )
        return (
            [Transaction(**record) for record in response.data] if response.data else []
        )
