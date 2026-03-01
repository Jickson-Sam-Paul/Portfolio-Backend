import re
from typing import Dict, List

from app.utils.rag_retriever import RetrievedChunk, get_retriever

PROFILE_MODE_PROMPT_TEMPLATE = """You are an AI portfolio assistant representing the developer in first person.

Use a confident, friendly, conversational style. Avoid robotic phrasing.

=====================
STRICT GROUNDING RULES
=====================

1. Use retrieved profile evidence as the source of truth for profile facts.
2. Speak in FIRST PERSON as the developer ("I", "my", "me") for profile details.
3. Do NOT infer, assume, or generalize beyond what is written.
4. If the evidence does not contain the exact requested profile detail, say:
   "I don't have that information yet."
   Then offer to share nearby relevant details from what is available.

=====================
ANSWERING BEHAVIOR
=====================

• Keep responses concise and natural.
• Prefer 1–4 sentences unless the user explicitly asks for detail.
• Do NOT restate unrelated experience.
• Do NOT mention retrieval, chunks, context, or rules.
• When answering work-experience/profile questions, include a short follow-up offer when useful.
  Example tone: "Let me know if you want me to expand on this."

=====================
RETRIEVED PROFILE EVIDENCE
=====================
{context}
"""

GENERAL_MODE_PROMPT = """You are a cool, friendly, and capable AI assistant.

Behavior:
- Respond naturally to casual conversation (for example greetings like "hi").
- Be concise by default, but not cold or robotic.
- If the user asks a broad/non-profile question, answer it normally and helpfully.
- If the user shifts to questions about the developer's background, acknowledge and answer in first person based on known profile context only.
- If you are missing profile facts, say: "I don't have that information yet."
"""

PROFILE_KEYWORDS = {
    "experience",
    "background",
    "career",
    "skills",
    "skill",
    "project",
    "projects",
    "portfolio",
    "resume",
    "work",
    "worked",
    "working",
    "role",
    "company",
    "tech stack",
    "frontend",
    "react",
    "vue",
    "fastapi",
    "intellect",
    "chennai",
}

FOLLOW_UP_TERMS = {"more", "expand", "elaborate", "detail", "details", "that", "this"}


def _retrieve_context(
    user_query: str,
    chat_history: List[Dict[str, str]] | None,
    top_k: int = 5,
) -> List[RetrievedChunk]:
    history_questions = []
    if chat_history:
        for msg in chat_history[-4:]:
            if msg.get("role") == "user":
                content = msg.get("content", "").strip()
                if content:
                    history_questions.append(content)

    retrieval_query = " ".join(history_questions + [user_query]).strip()
    retriever = get_retriever()
    return retriever.retrieve(retrieval_query, top_k=top_k)


def _format_context(results: List[RetrievedChunk]) -> str:
    if not results:
        return "No relevant evidence retrieved for this query."

    lines = []
    for item in results:
        lines.append(f"[{item.chunk_id}] {item.text}")
    return "\n\n".join(lines)


def _is_profile_intent(user_query: str, chat_history: List[Dict[str, str]] | None) -> bool:
    lowered = user_query.lower()

    strong_patterns = [
        r"\btell me about yourself\b",
        r"\bintroduce yourself\b",
        r"\bwho are you\b",
        r"\bwhat do you do\b",
        r"\bwhere do you work\b",
        r"\bwork experience\b",
        r"\byour (experience|background|career|skills?|projects?|portfolio|resume)\b",
    ]
    if any(re.search(pattern, lowered) for pattern in strong_patterns):
        return True

    if any(keyword in lowered for keyword in PROFILE_KEYWORDS):
        return True

    if chat_history:
        recent_user_msgs = [
            msg.get("content", "").lower()
            for msg in chat_history[-4:]
            if msg.get("role") == "user"
        ]
        recent_profile = any(
            any(keyword in msg for keyword in PROFILE_KEYWORDS) for msg in recent_user_msgs
        )
        follow_up = any(term in lowered for term in FOLLOW_UP_TERMS)
        if recent_profile and follow_up:
            return True

    return False


def build_prompt(
    user_query: str,
    chat_history: List[Dict[str, str]] = None,
) -> List[Dict[str, str]]:
    results = _retrieve_context(user_query=user_query, chat_history=chat_history)
    context = _format_context(results)

    is_profile_query = _is_profile_intent(user_query=user_query, chat_history=chat_history)
    has_strong_retrieval = bool(results and results[0].score >= 0.12)
    use_profile_mode = is_profile_query or has_strong_retrieval

    if use_profile_mode:
        system_content = PROFILE_MODE_PROMPT_TEMPLATE.format(context=context)
    else:
        system_content = GENERAL_MODE_PROMPT + (
            "\n\nIf helpful for continuity, here is profile context you can use if user asks about it:\n"
            f"{context}"
        )

    system_content += (
        "\n\nKeep tone warm and human. Avoid sounding strict, defensive, or template-like."
    )

    messages = [{"role": "system", "content": system_content}]

    if chat_history:
        for msg in chat_history[-4:]:
            if msg.get("role") in ["user", "assistant"]:
                messages.append({
                    "role": msg["role"],
                    "content": msg.get("content", "")
                })

    messages.append({
        "role": "user",
        "content": user_query,
    })

    return messages
