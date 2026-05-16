"""
Screen Channel — Awareness of what's on the user's screen.

Two layers of perception:
  1. Window title (always on, zero cost) — "VS Code - main.py" or "YouTube - How rockets work"
  2. Screenshot + LLM vision (optional, fires when title changes significantly)

This is separate from the webcam VisionChannel — that sees the physical
environment; this sees the digital activity. Different novelty rhythms:
your face is the same for hours, your screen changes every 30 seconds.

Privacy: respects excluded_apps and excluded_urls from config.
"""

from __future__ import annotations

import hashlib
import sys
from collections import deque

from src.models import ChannelInput


class ScreenChannel:
    """Monitors the active window and optionally captures screenshots."""

    def __init__(
        self,
        base_weight: float = 0.30,
        history_window: int = 10,
        screenshot_enabled: bool = True,
        min_novelty_for_screenshot: float = 0.5,
        excluded_apps: list[str] | None = None,
        excluded_urls: list[str] | None = None,
    ):
        self.base_weight = base_weight
        self.history_window = history_window
        self.screenshot_enabled = screenshot_enabled
        self.min_novelty_for_screenshot = min_novelty_for_screenshot
        self.excluded_apps = [a.lower() for a in (excluded_apps or [])]
        self.excluded_urls = [u.lower() for u in (excluded_urls or [])]
        self._title_history: deque[str] = deque(maxlen=history_window)
        self._available = False
        self._screenshot_available = False
        self._initialized = False

    def initialize(self) -> bool:
        """Check if window title reading and screenshots are possible."""
        if self._initialized:
            return self._available
        self._initialized = True

        if sys.platform == "win32":
            self._available = self._init_windows()
        else:
            self._available = self._init_cross_platform()

        if self._available:
            self._init_screenshot()

        return self._available

    def _init_windows(self) -> bool:
        """Try Windows-native window title reading."""
        try:
            import ctypes
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            if hwnd:
                print("   [Screen] Window title: ready (Win32 API)")
                return True
        except Exception:
            pass

        try:
            import pygetwindow
            win = pygetwindow.getActiveWindow()
            if win is not None:
                print("   [Screen] Window title: ready (pygetwindow)")
                return True
        except ImportError:
            pass
        except Exception:
            pass

        print("   [Screen] Window title: not available")
        return False

    def _init_cross_platform(self) -> bool:
        """Try cross-platform window title reading."""
        try:
            import pygetwindow
            win = pygetwindow.getActiveWindow()
            if win is not None:
                print("   [Screen] Window title: ready (pygetwindow)")
                return True
        except ImportError:
            print("   [Screen] pygetwindow not installed — screen awareness disabled")
        except Exception as e:
            print(f"   [Screen] Window title error: {e}")
        return False

    def _init_screenshot(self) -> None:
        """Check if screenshot capture is possible."""
        if not self.screenshot_enabled:
            return
        try:
            import mss  # noqa: F401
            self._screenshot_available = True
            print("   [Screen] Screenshots: ready (mss)")
        except ImportError:
            self._screenshot_available = False
            print("   [Screen] Screenshots: mss not installed — title-only mode")

    @property
    def is_available(self) -> bool:
        return self._available

    def get_active_window_title(self) -> str:
        """Get the title of the currently focused window."""
        if not self._available:
            return ""

        if sys.platform == "win32":
            title = self._get_title_win32()
            if title:
                return title

        try:
            import pygetwindow
            win = pygetwindow.getActiveWindow()
            if win and win.title:
                return win.title
        except Exception:
            pass

        return ""

    def _get_title_win32(self) -> str:
        """Windows-native: fast, no external dependency."""
        try:
            import ctypes
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
                return buf.value
        except Exception:
            pass
        return ""

    def _is_excluded(self, title: str) -> bool:
        """Check if this window should be ignored for privacy."""
        title_lower = title.lower()
        for app in self.excluded_apps:
            if app in title_lower:
                return True
        for url in self.excluded_urls:
            if url in title_lower:
                return True
        privacy_keywords = [
            "password", "1password", "lastpass", "bitwarden", "keepass",
            "private browsing", "incognito",
        ]
        return any(kw in title_lower for kw in privacy_keywords)

    def compute_novelty(self, title: str) -> float:
        """Compare current window title to recent history."""
        if not title:
            return 0.0

        normalized = self._normalize_title(title)

        if not self._title_history:
            self._title_history.append(normalized)
            return 0.5

        similarities = []
        for past in self._title_history:
            current_words = set(normalized.lower().split())
            past_words = set(past.lower().split())
            union = len(current_words | past_words)
            if union == 0:
                similarities.append(1.0)
            else:
                overlap = len(current_words & past_words)
                similarities.append(overlap / union)

        self._title_history.append(normalized)

        max_similarity = max(similarities) if similarities else 0.0
        return max(0.0, min(1.0, 1.0 - max_similarity))

    def _normalize_title(self, title: str) -> str:
        """Strip common suffixes like '- Google Chrome' or '- Visual Studio Code'."""
        noise = [
            " - Google Chrome", " - Mozilla Firefox", " - Microsoft Edge",
            " - Brave", " - Opera", " - Safari",
            " - Visual Studio Code", " - Cursor",
            " — Mozilla Firefox",
        ]
        for suffix in noise:
            if title.endswith(suffix):
                title = title[:-len(suffix)]
                break
        return title.strip()

    def capture_screenshot(self) -> bytes | None:
        """Capture a screenshot of the primary monitor."""
        if not self._screenshot_available:
            return None
        try:
            import mss
            import mss.tools
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                sct_img = sct.grab(monitor)
                from PIL import Image
                import io
                img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                img = img.resize((img.width // 2, img.height // 2))
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=60)
                return buf.getvalue()
        except ImportError:
            return None
        except Exception as e:
            print(f"   [Screen] Screenshot error: {e}")
            return None

    async def process(self, llm=None) -> ChannelInput:
        """Full screen pipeline: title → novelty → optional screenshot → description."""
        if not self._available:
            return ChannelInput(
                channel="screen",
                raw_content="",
                novelty=0.0,
                base_weight=self.base_weight,
                effective_weight=0.0,
                available=False,
            )

        title = self.get_active_window_title()

        if not title or self._is_excluded(title):
            return ChannelInput(
                channel="screen",
                raw_content="[private/excluded window]",
                novelty=0.0,
                base_weight=self.base_weight,
                effective_weight=0.0,
                available=True,
            )

        novelty = self.compute_novelty(title)
        normalized = self._normalize_title(title)
        content = f"Active window: {normalized}"

        if (novelty >= self.min_novelty_for_screenshot
                and self._screenshot_available and llm is not None):
            screenshot_desc = await self._describe_screenshot(llm)
            if screenshot_desc:
                content += f" | Screen content: {screenshot_desc}"

        effective_weight = self.base_weight * novelty

        novelty_label = "🔴 HIGH" if novelty > 0.6 else "🟡 MED" if novelty > 0.3 else "⚪ LOW"
        print(f"   [Screen] [{novelty_label} novelty={novelty:.2f}]: {content[:80]}")

        return ChannelInput(
            channel="screen",
            raw_content=content,
            novelty=novelty,
            base_weight=self.base_weight,
            effective_weight=effective_weight,
            available=True,
        )

    async def _describe_screenshot(self, llm) -> str:
        """Capture a screenshot and send to LLM vision for description."""
        screenshot_bytes = self.capture_screenshot()
        if not screenshot_bytes:
            return ""

        try:
            import base64
            b64 = base64.b64encode(screenshot_bytes).decode("utf-8")

            from openai import AsyncOpenAI
            import os
            client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Describe what's on this screen in 1-2 concise sentences. "
                                "Focus on the main activity: what app, what content, what "
                                "the user seems to be doing. Be specific about any visible "
                                "text, code, video content, or game UI."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
                        },
                    ],
                }],
                max_tokens=150,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"   [Screen] Screenshot description failed: {e}")
            return ""
