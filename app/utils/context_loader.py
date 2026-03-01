from pathlib import Path

PROFILE_FILE = Path("app/data/profile.txt")

_cached_context: str | None = None


def load_full_context() -> str:
    """
    Load the developer profile context from a single file.
    Cached after first load.
    """
    global _cached_context

    if _cached_context is not None:
        return _cached_context

    text = PROFILE_FILE.read_text(encoding="utf-8").strip()

    # Add one identity anchor line for extra grounding
    _cached_context = f"DEVELOPER PROFILE:\n{text}"
    return _cached_context