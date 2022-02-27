dependencies = ["torch"]

import sys
from src.silero import (
    silero_stt,
    silero_tts,
    silero_te,
)

__all__ = [
    "silero_stt",
    "silero_tts",
    "silero_te",
]

sys.path.append("src/silero")
