"""
led_status.py — GPIO LED Status Indicators for Raspberry Pi 5.

Controls 3x physical status LEDs connected via GPIO pins:
    - LED 1 (GPIO 22 / Pin 15): Power & System Active Indicator (Solid ON when streaming)
    - LED 2 (GPIO 23 / Pin 16): Enhancement Mode Indicator (ON = GTCRN Filtered, OFF = Raw Bypass)
    - LED 3 (GPIO 24 / Pin 18): Audio Activity / Peak Signal / Impulse Warning Indicator

Uses gpiozero for Raspberry Pi 5 compatibility (lgpio backend).
Fails gracefully with a no-op stub on non-Pi platforms.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


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
        self._led_sys = None
        self._led_mode = None
        self._led_act = None

        if not self.enabled:
            return

        try:
            from gpiozero import LED  # type: ignore

            self._led_sys = LED(led_sys_pin)
            self._led_mode = LED(led_mode_pin)
            self._led_act = LED(led_act_pin)

            logger.info(
                "GPIO LEDs initialized: SYS=GPIO%d, MODE=GPIO%d, ACT=GPIO%d",
                led_sys_pin,
                led_mode_pin,
                led_act_pin,
            )
        except Exception as exc:
            logger.warning("GPIO LEDs not available (%s) — running in headless mode", exc)
            self.enabled = False

    def set_system_active(self, active: bool = True) -> None:
        """Set System Status LED (LED 1 - GPIO 22)."""
        if self._led_sys:
            if active:
                self._led_sys.on()
            else:
                self._led_sys.off()

    def set_enhancement_mode(self, is_enhanced: bool) -> None:
        """Set Enhancement Mode LED (LED 2 - GPIO 23)."""
        if self._led_mode:
            if is_enhanced:
                self._led_mode.on()
            else:
                self._led_mode.off()

    def trigger_activity(self, duration_sec: float = 0.05) -> None:
        """Pulse Activity / Warning LED (LED 3 - GPIO 24)."""
        if self._led_act:
            self._led_act.on()

    def set_activity(self, active: bool) -> None:
        """Set Activity / Warning LED state directly."""
        if self._led_act:
            if active:
                self._led_act.on()
            else:
                self._led_act.off()

    def close(self) -> None:
        """Turn off all LEDs upon shutdown."""
        for led in (self._led_sys, self._led_mode, self._led_act):
            if led:
                try:
                    led.off()
                    led.close()
                except Exception:
                    pass
