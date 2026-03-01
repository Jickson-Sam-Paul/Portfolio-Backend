from typing import AsyncIterator
from fastapi import APIRouter
from fastapi.responses import StreamingResponse, Response
from app.api.prompt_builder import build_prompt
from app.api.llm import get_llm
from app.models.chat import ChatRequest

router = APIRouter()

_llm = None

def get_llm_instance():
    global _llm
    if _llm is None:
        try:
            _llm = get_llm()
        except Exception as e:
            print(f"Failed to initialize LLM: {e}")
            return None
    return _llm


async def generate_response(
    user_query: str,
    chat_history: list
) -> AsyncIterator[str]:
    try:
        messages = build_prompt(
            user_query=user_query,
            chat_history=chat_history
        )
        
        llm = get_llm_instance()
        if llm is None:
            yield "The AI assistant is currently unavailable."
            return
        
        async for chunk in llm.stream_completion(messages):
            yield chunk
            
    except Exception as e:
        yield "The AI assistant is currently unavailable."


@router.options("/chat")
async def chat_options():
    return Response(status_code=200)


@router.post("/chat")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    messages = request.messages
    if not messages:
        async def empty_response():
            yield ""
        return StreamingResponse(
            empty_response(),
            media_type="text/plain",
        )

    user_query = ""
    for msg in reversed(messages):
        if msg.role == "user":
            user_query = msg.content
            break
    
    if not user_query:
        async def empty_response():
            yield ""
        return StreamingResponse(
            empty_response(),
            media_type="text/plain",
        )
    
    chat_history = [
        {"role": msg.role, "content": msg.content}
        for msg in messages
    ]
    
    return StreamingResponse(
        generate_response(user_query, chat_history),
        media_type="text/plain",
    )


@router.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "portfolio-backend"
    }
