"""
Heartbeat — The Serendipity Clock.

A random timer that fires at unpredictable intervals, creating the hidden
causation that produces the appearance of spontaneous thought.

ADAPTIVE TEMPO: The heartbeat speeds up when novelty is high (lots happening,
pay attention!) and slows down when things are static (nothing changing,
back off). Uses a rolling average of recent novelty to avoid jerky transitions.
This matches how a real friend's attention works — they lean in when things
get interesting and zone out when nothing's happening.

Supports both single-fire (testing) and continuous loop (live companion mode).
"""

from __future__ import annotations

import asyncio
import random
from collections import deque
from datetime import datetime
from typing import Callable, Awaitable

from src.models import ContextSnapshot


class Heartbeat:
    def __init__(
        self,
        min_minutes: int = 1,
        max_minutes: int = 10,
        adaptive: bool = True,
        fast_min: float = 1.0,
        fast_max: float = 3.0,
        slow_min: float = 7.0,
        slow_max: float = 15.0,
        novelty_window: int = 3,
    ):
        self.base_min = min_minutes
        self.base_max = max_minutes
        self.min_minutes = min_minutes
        self.max_minutes = max_minutes
        self.adaptive = adaptive
        self.fast_min = fast_min
        self.fast_max = fast_max
        self.slow_min = slow_min
        self.slow_max = slow_max
        self._novelty_history: deque[float] = deque(maxlen=novelty_window)
        self._running = False
        self.beat_count = 0
        self._remaining_seconds = 0
        self._backoff_beats = 0
        self._current_tempo = "normal"

    def next_interval_seconds(self) -> int:
        """Pick a random interval within the current adaptive range, return as seconds."""
        interval = random.uniform(self.min_minutes, self.max_minutes)
        return int(interval * 60)

    def adjust_tempo(self, novelty: float) -> None:
        """Adjust heartbeat speed based on observed novelty.

        Uses a rolling average to smooth transitions. High novelty = faster
        beats (lean in), low novelty = slower beats (zone out). The randomness
        within the range is preserved so timing stays unpredictable.
        """
        if not self.adaptive:
            return

        self._novelty_history.append(novelty)
        avg_novelty = sum(self._novelty_history) / len(self._novelty_history)

        if avg_novelty >= 0.65:
            self.min_minutes = self.fast_min
            self.max_minutes = self.fast_max
            new_tempo = "fast"
        elif avg_novelty >= 0.35:
            blend = (avg_novelty - 0.35) / 0.30
            self.min_minutes = self.slow_min + blend * (self.fast_min - self.slow_min)
            self.max_minutes = self.slow_max + blend * (self.fast_max - self.slow_max)
            new_tempo = "normal"
        else:
            self.min_minutes = self.slow_min
            self.max_minutes = self.slow_max
            new_tempo = "slow"

        if new_tempo != self._current_tempo:
            self._current_tempo = new_tempo
            labels = {"fast": "⚡ FAST", "normal": "💓 NORMAL", "slow": "🐌 SLOW"}
            print(f"   {labels[new_tempo]} — Heartbeat adapted: "
                  f"{self.min_minutes:.0f}–{self.max_minutes:.0f} min "
                  f"(avg novelty: {avg_novelty:.2f})")

    @property
    def tempo_info(self) -> str:
        """Human-readable tempo status."""
        avg = sum(self._novelty_history) / len(self._novelty_history) if self._novelty_history else 0.5
        return f"{self._current_tempo} ({self.min_minutes:.0f}–{self.max_minutes:.0f}m, novelty avg: {avg:.2f})"

    async def start(self, on_beat: Callable[[ContextSnapshot], Awaitable[None]]) -> None:
        """Start the continuous heartbeat loop. Calls on_beat every time the timer fires."""
        self._running = True
        mode = "adaptive" if self.adaptive else "fixed"
        print(f"   💓 Heartbeat range: {self.min_minutes:.0f}–{self.max_minutes:.0f} minutes ({mode})")

        while self._running:
            interval = self.next_interval_seconds()
            self._remaining_seconds = interval
            minutes_display = interval / 60
            print(f"\n   💓 Next heartbeat in {minutes_display:.1f} minute(s)... [{self._current_tempo}]")

            while self._remaining_seconds > 0 and self._running:
                sleep_chunk = min(1, self._remaining_seconds)
                await asyncio.sleep(sleep_chunk)
                self._remaining_seconds -= sleep_chunk

            if not self._running:
                break

            if self._backoff_beats > 0:
                self._backoff_beats -= 1
                print(f"   💤 Heartbeat suppressed (backing off, {self._backoff_beats} skips remaining)")
                continue

            self.beat_count += 1
            ctx = ContextSnapshot(
                timestamp=datetime.now(),
                heartbeat_id=f"hb-{self.beat_count:04d}",
            )
            print(f"\n{'═' * 70}")
            print(f"💓 HEARTBEAT #{self.beat_count} fired at {ctx.timestamp.strftime('%H:%M:%S')}")
            print(f"{'═' * 70}")
            await on_beat(ctx)

    def stop(self) -> None:
        self._running = False
        self._remaining_seconds = 0

    def backoff(self, beats: int = 2) -> None:
        """Skip the next N heartbeats — user said 'not now'."""
        self._backoff_beats = beats
        print(f"   💤 Backing off for {beats} heartbeat(s)")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def time_until_next(self) -> int:
        return max(0, self._remaining_seconds)

    async def fire_once(self, seed_topic: str) -> ContextSnapshot:
        """Fire a single heartbeat immediately with a given seed topic (for testing)."""
        self.beat_count += 1
        ctx = ContextSnapshot(
            timestamp=datetime.now(),
            heartbeat_id=f"hb-{self.beat_count:04d}",
            seed_topic=seed_topic,
        )
        print(f"\n💓 HEARTBEAT #{self.beat_count} fired at {ctx.timestamp.strftime('%H:%M:%S')}")
        print(f"   Seed topic: \"{seed_topic}\"")
        return ctx
