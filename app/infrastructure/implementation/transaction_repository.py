from typing import Optional
from app.infrastructure.interfaces.transaction_repository import ITransactionRepository
from app.domain.transaction import Transaction, TransactionType
from app.core.supabase import create_client, get_supabase
from uuid import UUID
import os

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


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

    def get_user_transactions(self, user_id: str):
        response = (
            supabase.table("transactions").select("*").eq("user_id", user_id).execute()
        )
        data = response.data
        return data
