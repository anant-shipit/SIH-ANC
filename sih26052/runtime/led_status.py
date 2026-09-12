"""
led_status.py — GPIO LED Status Indicators for Raspberry Pi 5.

Controls 3x physical status LEDs connected via GPIO pins:
    - LED 1 (GPIO 22 / Pin 15): Power & System Active Indicator (Solid ON when streaming)
    - LED 2 (GPIO 23 / Pin 16): Enhancement Mode Indicator (ON = GTCRN Filtered, OFF = Raw Bypass)
    - LED 3 (GPIO 24 / Pin 18): Audio Activity / Peak Signal / Impulse Warning Indicator

Supports dual backends on Raspberry Pi 5:
    1. gpiozero (lgpio backend) with automatic pin instance caching to avoid "GPIO busy" errors
    2. Native kernel pinctrl fallback (bypasses all busy locks directly on the RP1 chip)
Fails gracefully with a no-op stub on non-Pi platforms.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Process-wide cache of allocated gpiozero LED instances to prevent "GPIO busy" re-allocation crashes
_GLOBAL_PIN_REGISTRY: Dict[int, Any] = {}


def _get_or_create_led(pin: int) -> Optional[Any]:
    """Retrieve existing open gpiozero LED or allocate a new one safely."""
    global _GLOBAL_PIN_REGISTRY
    if pin in _GLOBAL_PIN_REGISTRY:
        obj = _GLOBAL_PIN_REGISTRY[pin]
        try:
            if not getattr(obj, "closed", False):
                return obj
        except Exception:
            pass

    try:
        from gpiozero import LED  # type: ignore
        led = LED(pin)
        _GLOBAL_PIN_REGISTRY[pin] = led
        return led
    except Exception as exc:
        logger.debug("gpiozero allocation on pin %d failed (%s)", pin, exc)
        return None


class LEDStatus:
    """Manager for the 3x physical status LEDs on Raspberry Pi 5."""

    def __init__(
        self,
        led_sys_pin: int = 22,
        led_mode_pin: int = 23,
        led_act_pin: int = 24,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.sys_pin = led_sys_pin
        self.mode_pin = led_mode_pin
        self.act_pin = led_act_pin
        self._led_sys = None
        self._led_mode = None
        self._led_act = None

        if not self.enabled:
            return

        # 1. Attempt allocation with gpiozero
        self._led_sys = _get_or_create_led(led_sys_pin)
        self._led_mode = _get_or_create_led(led_mode_pin)
        self._led_act = _get_or_create_led(led_act_pin)

        # 2. Check if pinctrl is available on Pi 5 as a universal direct-kernel fallback
        self._has_pinctrl = shutil.which("pinctrl") is not None

        if self._led_sys is not None:
            logger.info(
                "GPIO LEDs initialized via gpiozero: SYS=GPIO%d, MODE=GPIO%d, ACT=GPIO%d",
                led_sys_pin,
                led_mode_pin,
                led_act_pin,
            )
            self.enabled = True
        elif self._has_pinctrl:
            logger.info(
                "Using native pinctrl backend for GPIO LEDs: SYS=GPIO%d, MODE=GPIO%d, ACT=GPIO%d",
                led_sys_pin,
                led_mode_pin,
                led_act_pin,
            )
            self.enabled = True
        else:
            logger.warning("No GPIO LED backend available — running in headless mode")
            self.enabled = False

    def _drive_pin(self, pin: int, led_obj: Any, active: bool) -> None:
        """Drive a physical pin high or low using gpiozero or pinctrl fallback."""
        if not self.enabled:
            return

        # Try gpiozero object first
        if led_obj is not None:
            try:
                if active:
                    led_obj.on()
                else:
                    led_obj.off()
                return
            except Exception:
                pass

        # Fallback to direct pinctrl (never errors with "GPIO busy")
        if self._has_pinctrl:
            try:
                subprocess.run(
                    ["pinctrl", "set", str(pin), "op", "dh" if active else "dl"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            except Exception:
                pass

    def set_system_active(self, active: bool = True) -> None:
        """Set System Status LED (LED 1 - GPIO 22)."""
        self._drive_pin(self.sys_pin, self._led_sys, active)

    def set_enhancement_mode(self, is_enhanced: bool) -> None:
        """Set Enhancement Mode LED (LED 2 - GPIO 23)."""
        self._drive_pin(self.mode_pin, self._led_mode, is_enhanced)

    def trigger_activity(self, duration_sec: float = 0.05) -> None:
        """Pulse Activity / Warning LED (LED 3 - GPIO 24)."""
        self._drive_pin(self.act_pin, self._led_act, True)

    def set_activity(self, active: bool) -> None:
        """Set Activity / Warning LED state directly."""
        self._drive_pin(self.act_pin, self._led_act, active)

    def close(self) -> None:
        """Turn off all LEDs upon shutdown."""
        self.set_system_active(False)
        self.set_enhancement_mode(False)
        self.set_activity(False)
