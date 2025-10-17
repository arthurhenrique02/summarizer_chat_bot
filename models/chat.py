from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    text: str = Field(..., description="The chat message text")
