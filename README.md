# SavvyAPI 💡💰

Smart Personal Finance API built with FastAPI and Clean Architecture.  
Features Supabase for storage and LangChain for Retrieval-Augmented Generation (RAG) chatbot functionality.

---

## 🚀 Features

- **User Management**: Register, upload profile image, update, and delete users.
- **Password Handling**: Secure password storage and password reset support.
- **Transactions**: Auto-categorized using LLM (e.g., "Spent 50 EGP on lunch").
- **RAG Chatbot**: Ask finance-related questions based on user data and uploaded PDFs.
- **Clean Architecture**: Decoupled layers for better maintainability.
- **Repository Desgin Pattern**: Handle and manage all the database operation at one place.
- **Interactive Docs**: OpenAPI-powered documentation via Swagger and ReDoc.

---

## 📦 Getting Started

### Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org) (optional but recommended)
- `.env` file in root with:

```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_service_key
API_KEY=your_llm_provider_key
JWT_SECRET=your_jwt_secret
