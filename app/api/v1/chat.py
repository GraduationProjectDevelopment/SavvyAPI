from fastapi import APIRouter
from app.models.chat_dto import ChatRequest
from app.services.implementation.chat_service import ChatService
from app.services.interfaces.chat_service import IChatService


router = APIRouter()
chat_service: IChatService = ChatService()


@router.post("/", summary="Send a message and get AI response")
def chat(req: ChatRequest):
    reply = chat_service.get_response(req.user_id, req.message)
    return {"response": reply}
