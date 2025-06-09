from app.infrastructure.interfaces.time_series_repository import ITimeSeriesRepository
from app.models.time_series_dto import (
    TimeSeriesPredictionResponse,
    MonthlyExpense,
    ExpenseBreakdown,
)
from app.core.supabase import get_supabase
from fastapi import HTTPException
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from tensorflow.keras.models import load_model


class TimeSeriesRepository(ITimeSeriesRepository):
    def __init__(self):
        self.supabase = get_supabase()
        self.model = None
        self.scaler = None
        self.load_ml_models()

    def load_ml_models(self):
        base_path = os.path.join(os.path.dirname(__file__), "../../../core/TimeSeries")
        model_path = os.path.abspath(os.path.join(base_path, "lstm.h5"))
        scaler_path = os.path.abspath(os.path.join(base_path, "scaler.save"))

        try:
            self.model = load_model(model_path)
            self.scaler = joblib.load(scaler_path)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Model loading failed: {str(e)}"
            )

    def get_next_month_prediction(self, user_id: str) -> TimeSeriesPredictionResponse:
        df = self._fetch_data(user_id)
        df["created_at"] = (
            pd.to_datetime(df["created_at"]).dt.to_period("M").dt.to_timestamp()
        )

        category_map = (
            self.supabase.from_("categories")
            .select("category_id, category_name")
            .execute()
            .data
        )
        cat_df = pd.DataFrame(category_map)
        df = df.merge(cat_df, on="category_id", how="left").dropna(
            subset=["category_name"]
        )

        monthly_expenses = (
            df.groupby(["created_at", "category_name"])["amount"]
            .sum()
            .unstack(fill_value=0)
            .sort_index()
        )
        expected = [
            "Education",
            "Entertainment",
            "Fashion",
            "Food",
            "Lifestyle",
            "Transportation",
            "Health",
        ]
        X = pd.DataFrame(index=monthly_expenses.index)
        for col in expected:
            X[col] = monthly_expenses.get(col, 0)
        X["Expenses"] = X[expected].sum(axis=1)

        # Drop current month if not complete
        today = pd.Timestamp(datetime.today().date())
        if X.index[-1].month == today.month:
            X = X[:-1]

        if len(X) < 3:
            raise HTTPException(
                status_code=400, detail="Need at least 3 months of data"
            )

        last_3 = X.tail(3)
        scaled = self.scaler.transform(last_3[expected])
        input_data = np.expand_dims(scaled, axis=0)
        pred = float(self.model.predict(input_data)[0, 0])

        next_month = (last_3.index.max() + pd.offsets.MonthBegin(1)).strftime(
            "%Y-%m-%d"
        )
        next_month_label = (last_3.index.max() + pd.offsets.MonthBegin(1)).strftime(
            "%Y-%m"
        )

        history = []
        for _, row in last_3.iterrows():
            history.append(
                MonthlyExpense(
                    month=row.name.strftime("%Y-%m"),
                    total_expense=float(row["Expenses"]),
                    breakdown=ExpenseBreakdown(
                        **{col: float(row[col]) for col in expected}
                    ),
                )
            )

        return TimeSeriesPredictionResponse(
            user_id=user_id,
            date=next_month,
            month=next_month_label,
            predicted_total_expense=round(pred, 2),
            history=history,
        )

    def _fetch_data(self, user_id: str):
        response = (
            self.supabase.from_("transactions")
            .select("category_id, amount, created_at")
            .eq("user_id", user_id)
            .eq("transaction_type", "Expense")
            .order("created_at")
            .execute()
        )
        if not response.data:
            raise HTTPException(status_code=404, detail="No expense data")
        return pd.DataFrame(response.data)
