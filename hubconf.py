dependencies = ["torch"]

import sys
from src.silero import (
    silero_stt,
    silero_tts,
    silero_te,
    silero_denoise,
)

__all__ = [
    "silero_stt",
    "silero_tts",
    "silero_te",
    "silero_denoise",
]

sys.path.append("src/silero")
