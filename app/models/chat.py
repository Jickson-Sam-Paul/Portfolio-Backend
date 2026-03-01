from typing import List
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., description="List of chat messages")

