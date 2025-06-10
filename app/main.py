from fastapi import FastAPI
from dotenv import load_dotenv

from app.api.v1 import user, chat, transactions, budget, goal ,time_series

load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Personal Finance Management App",
    version="1.0",
    description="Mobile Application help users Track spending, set goals, get insights.",
)

# Router prefix
api_v1_prefix = "/api/v1"

# Include API routers
app.include_router(user.router, prefix=api_v1_prefix + "/users", tags=["Users"])
app.include_router(chat.router, prefix=api_v1_prefix + "/chat", tags=["Chat"])
app.include_router(
    transactions.router, prefix=api_v1_prefix + "/transactions", tags=["Transactions"]
)

app.include_router(budget.router, prefix=api_v1_prefix + "/budgets", tags=["Budgets"])

app.include_router(time_series.router, prefix="/forecast", tags=["Forecasting"])

app.include_router(goal.router, prefix=api_v1_prefix + "/goals", tags=["Goals"])
