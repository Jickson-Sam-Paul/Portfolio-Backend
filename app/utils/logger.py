# app/utils/logger.py
import json
from datetime import datetime

def log_chat(user_query: str, ai_response: str):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_query": user_query,
        "ai_response": ai_response
    }

    with open("chat_logs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")