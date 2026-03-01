from typing import AsyncIterator, List, Dict, Any, Optional
from groq import Groq
import os

class GroqLLM:
    def __init__(self, model: str = "llama-3.1-8b-instant", api_key: Optional[str] = None):
        self.model = model
        api_key = api_key or os.getenv("GROQ_API_KEY")
        
        if not api_key:
            raise ValueError(
                "Groq API key not provided. Set GROQ_API_KEY environment variable."
            )
        
        self.client = Groq(api_key=api_key)

    async def stream_completion(
        self,
        messages: List[Dict[str, str]]
    ) -> AsyncIterator[str]:
        try:
            # Need Clarification
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                temperature=0.7,
                max_tokens=1024
            )

            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield delta.content

        except Exception as e:
            error_msg = f"Error in LLM call: {str(e)}"
            print(error_msg)
            yield "The AI assistant is currently unavailable."


def get_llm(model: str = "llama-3.1-8b-instant") -> GroqLLM:
    return GroqLLM(model=model)

