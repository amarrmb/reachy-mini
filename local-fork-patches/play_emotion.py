import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Initialize emotion library
try:
    from reachy_mini.motion.recorded_move import RecordedMoves
    from reachy_mini_conversation_app.dance_emotion_moves import EmotionQueueMove

    # Note: huggingface_hub automatically reads HF_TOKEN from environment variables
    RECORDED_MOVES = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
    EMOTION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Emotion library not available: {e}")
    RECORDED_MOVES = None
    EMOTION_AVAILABLE = False


def get_available_emotions_and_descriptions() -> str:
    """Get formatted list of available emotions with descriptions."""
    if not EMOTION_AVAILABLE:
        return "Emotions not available"

    try:
        emotion_names = RECORDED_MOVES.list_moves()
        output = "Available emotions:\n"
        for name in emotion_names:
            description = RECORDED_MOVES.get(name).description
            output += f" - {name}: {description}\n"
        return output
    except Exception as e:
        return f"Error getting emotions: {e}"



_EMOTION_ALIASES = {
    "happy": "cheerful1", "joyful": "cheerful1", "cheerful": "cheerful1",
    "sad": "sad2", "upset": "sad2", "down": "sad2",
    "curious": "curious1", "interested": "curious1",
    "scared": "scared1", "afraid": "scared1", "frightened": "scared1",
    "shy": "shy1", "embarrassed": "shy1",
    "angry": "displeased1", "annoyed": "displeased1",
    "proud": "proud1",
    "surprise": "surprised1", "surprised": "surprised1",
    "sleepy": "sleepy1", "tired": "sleepy1",
    "yes": "understanding2", "agree": "understanding2", "nod": "understanding2",
    "oops": "oops1", "sorry": "oops1",
    "wave": "hello1", "hello": "hello1", "hi": "hello1",
    "thinking": "inquiring1", "think": "inquiring1",
    "relief": "relief1", "relieved": "relief1",
}

def _resolve_emotion(name, available):
    """Map friendly emotion names to library entries via aliases + fuzzy match."""
    if not name: return None
    n = name.strip().lower()
    if n in available: return n
    if name in available: return name
    # Alias
    if n in _EMOTION_ALIASES and _EMOTION_ALIASES[n] in available:
        return _EMOTION_ALIASES[n]
    # Suffix-numbered match: happy -> cheerful1, curious -> curious1
    for cand in available:
        if cand.lower().rstrip("0123456789") == n: return cand
        if cand.lower().startswith(n): return cand
    return None
class PlayEmotion(Tool):
    """Play a pre-recorded emotion."""

    name = "play_emotion"
    description = "Play a pre-recorded emotion"
    parameters_schema = {
        "type": "object",
        "properties": {
            "emotion": {
                "type": "string",
                "description": f"""Name of the emotion to play.
                                    Here is a list of the available emotions:
                                    {get_available_emotions_and_descriptions()}
                                    """,
            },
        },
        "required": ["emotion"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Play a pre-recorded emotion."""
        if not EMOTION_AVAILABLE:
            return {"error": "Emotion system not available"}

        emotion_name = kwargs.get("emotion")
        if not emotion_name:
            return {"error": "Emotion name is required"}

        logger.info("Tool call: play_emotion emotion=%s", emotion_name)

        try:
            emotion_names = RECORDED_MOVES.list_moves()
            resolved = _resolve_emotion(emotion_name, emotion_names)
            if not resolved:
                return {"error": f"Unknown emotion '{emotion_name}'. Available: {emotion_names}"}
            if resolved != emotion_name:
                logger.info("Resolved %r -> %r", emotion_name, resolved)
            emotion_name = resolved

            # Add emotion to queue
            movement_manager = deps.movement_manager
            emotion_move = EmotionQueueMove(emotion_name, RECORDED_MOVES)
            movement_manager.queue_move(emotion_move)

            return {"status": "queued", "emotion": emotion_name}

        except Exception as e:
            logger.exception("Failed to play emotion")
            return {"error": f"Failed to play emotion: {e!s}"}
