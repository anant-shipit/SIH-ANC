from .ab_switch import ABSwitch
from .enhancer import StreamingEnhancer
from .impulse_gate import ImpulseGate
from .led_status import LEDStatus
from .nlms import NLMSFilter
from .ola import OverlapAdd

__all__ = [
    "ABSwitch",
    "AudioLoop",
    "StreamingEnhancer",
    "ImpulseGate",
    "LEDStatus",
    "NLMSFilter",
    "OverlapAdd",
]


def __getattr__(name: str):
    if name == "AudioLoop":
        from .audio_loop import AudioLoop
        return AudioLoop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

