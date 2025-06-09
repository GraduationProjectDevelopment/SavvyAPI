from app.services.interfaces.chat_service import IChatService
from app.core.rag_model import get_chat_response


class ChatService(IChatService):
    def get_response(self, user_id: str, message: str) -> str:
        result = get_chat_response(message, user_id)
        if "error" in result:
            return f"Error: {result['error']}"
        return result["answer"]
