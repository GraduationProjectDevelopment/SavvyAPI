import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    SUPABASE_URL: str = os.getenv("SUPABASE_URL")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY")
    JWT_SECRET: str = os.getenv("JWT_SECRET", "secret")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    UPLOAD_FOLDER: str = os.path.join(os.path.dirname(__file__), "static")


settings = Settings()
