"""
Incubation Queue — Where ideas percolate before emerging as insights.

Real creativity doesn't happen instantly. Ideas sit in the back of your mind,
marinating alongside new experiences, until suddenly the connection clicks.
This module simulates that process:

  1. Low-scoring chains (between incubation_threshold and fire_threshold) enter the queue
  2. Every hour, incubating chains are re-scored against the CURRENT context
  3. If a chain's score rises above fire_threshold with new context, it "hatches"
  4. Every 24 hours, the best incubated ideas are synthesized into a reflection

The result: "Hey, I've been thinking about something all day..." — and it
actually HAS been thinking about it. The incubation period is real.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime

from src.config.llm_adapter import LLMAdapter
from src.memory.store import MemoryStore, StoredChain
from src.models import AssociationChain, AssociationNode, ContextSnapshot


DAILY_REFLECTION_PROMPT = """You are a creative companion who has been thinking about ideas all day. Some associations you noticed earlier didn't quite click at the time, but now — looking at them together with what happened today — you see deeper connections.

Here are the most interesting ideas that have been percolating:

{incubated_ideas}

The user's current context: "{current_context}"

Write a 3-5 sentence "daily reflection" that synthesizes the BEST connection you see between these incubated ideas. Sound like a thoughtful friend sharing an insight that's been building all day. Start naturally — like "You know, I've been thinking about something..." or "Something clicked for me just now..."

Be specific. Name the actual topics. Draw a non-obvious connection between at least two of the incubated ideas. This should feel like a genuine "aha" moment, not a summary.

Do NOT use markdown, asterisks, or any formatting. Just plain conversational text."""


class IncubationQueue:
    """Manages the lifecycle of incubating ideas.

    Chains enter when they score between incubation_threshold and fire_threshold.
    They're periodically re-scored against new context. Chains that ripen
    above the fire threshold are "promoted" and delivered as interjections.
    Chains that never ripen after max_age_hours are expired gracefully.
    """

    def __init__(
        self,
        memory: MemoryStore,
        llm: LLMAdapter,
        rescore_interval_minutes: int = 60,
        max_age_hours: int = 24,
        max_rescores: int = 5,
        daily_reflection_hour: int = 22,
    ):
        self.memory = memory
        self.llm = llm
        self.rescore_interval = rescore_interval_minutes * 60
        self.max_age_hours = max_age_hours
        self.max_rescores = max_rescores
        self.daily_reflection_hour = daily_reflection_hour
        self._running = False
        self._last_reflection_date: str | None = None
        self._on_promotion = None
        self._current_context: str = ""

    def set_context(self, context: str) -> None:
        """Update the current context for re-scoring."""
        self._current_context = context

    async def start(self, on_promotion=None) -> None:
        """Start the background incubation loop.

        on_promotion is called when an incubating chain ripens and should be
        delivered as an interjection. Signature: async def callback(StoredChain)
        """
        self._running = True
        self._on_promotion = on_promotion

        print(f"   [Incubation] Queue active — rescore every {self.rescore_interval // 60}m, "
              f"expire after {self.max_age_hours}h")

        while self._running:
            await asyncio.sleep(self.rescore_interval)
            if not self._running:
                break

            await self._rescore_cycle()
            await self._check_daily_reflection()

    def stop(self) -> None:
        self._running = False

    def incubate(
        self,
        chain_id: str,
        seed_topic: str,
        endpoint_topic: str,
        chain_summary: str,
        domains: list[str],
        domain_crossings: int,
        score: float,
        embedding=None,
        context: str = "",
    ) -> None:
        """Add a chain to the incubation queue."""
        self.memory.store_chain(
            chain_id=chain_id,
            seed_topic=seed_topic,
            endpoint_topic=endpoint_topic,
            chain_summary=chain_summary,
            domains=domains,
            domain_crossings=domain_crossings,
            score=score,
            embedding=embedding,
            context=context,
            status="incubating",
        )

    @property
    def queue_size(self) -> int:
        """Number of currently incubating chains."""
        return len(self.memory.get_incubating())

    async def _rescore_cycle(self) -> None:
        """Re-evaluate all incubating chains against current context."""
        incubating = self.memory.get_incubating()
        if not incubating:
            return

        now = time.time()
        promoted = 0
        expired = 0

        for chain in incubating:
            age_hours = (now - chain.timestamp) / 3600

            if age_hours > self.max_age_hours or chain.rescore_count >= self.max_rescores:
                self.memory.update_chain_status(chain.id, "expired")
                expired += 1
                continue

            new_score = await self._rescore_chain(chain)

            if new_score is not None and new_score >= 0.45:
                self.memory.update_chain_status(chain.id, "promoted", new_score)
                promoted += 1
                if self._on_promotion:
                    await self._on_promotion(chain, new_score)
            else:
                self.memory.update_chain_status(
                    chain.id, "incubating",
                    new_score if new_score else chain.score,
                )

        if promoted or expired:
            remaining = len(incubating) - promoted - expired
            print(f"   [Incubation] Cycle: {promoted} promoted, {expired} expired, "
                  f"{remaining} still incubating")

    async def _rescore_chain(self, chain: StoredChain) -> float | None:
        """Re-score a single chain against the current context.

        Uses bridgeability as the primary re-evaluation metric since
        context relevance is what changes over time.
        """
        if not self._current_context:
            return None

        prompt = f"""A creative association chain went from "{chain.seed_topic}" to "{chain.endpoint_topic}" through:
{chain.chain_summary}

The user is NOW focused on: "{self._current_context}"

Rate on 0.0 to 1.0: How relevant and interesting is this connection RIGHT NOW given what the user is doing?

0.0 = completely irrelevant to their current activity
0.3 = tangentially related
0.5 = moderately relevant — could spark an interesting thought
0.7 = highly relevant — this connection feels timely
1.0 = perfectly relevant — this is exactly what they need to hear right now

Return ONLY a single number."""

        try:
            new_bridge_score = await self.llm.generate_float(prompt)
            original_base = chain.score * 0.4
            new_context_weight = new_bridge_score * 0.6
            return original_base + new_context_weight
        except Exception:
            return None

    async def _check_daily_reflection(self) -> None:
        """Check if it's time for the daily synthesis."""
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")

        if self._last_reflection_date == today_str:
            return

        if now.hour >= self.daily_reflection_hour:
            self._last_reflection_date = today_str
            await self._generate_daily_reflection()

    async def _generate_daily_reflection(self) -> None:
        """Synthesize the day's best incubated ideas into a reflection.

        Looks at all chains that were incubated or promoted today and
        asks the LLM to find the deepest connection between them.
        """
        incubating = self.memory.get_incubating(limit=10)
        recent_fired = self.memory.get_recent(limit=5, status="fired")
        promoted = self.memory.get_recent(limit=5, status="promoted")

        all_ideas = incubating + promoted + recent_fired
        if len(all_ideas) < 2:
            return

        all_ideas.sort(key=lambda c: c.score, reverse=True)
        top_ideas = all_ideas[:6]

        ideas_text = "\n".join(
            f"  - \"{idea.seed_topic}\" → \"{idea.endpoint_topic}\" "
            f"(through: {idea.chain_summary[:80]}) [score: {idea.score:.2f}]"
            for idea in top_ideas
        )

        prompt = DAILY_REFLECTION_PROMPT.format(
            incubated_ideas=ideas_text,
            current_context=self._current_context or "general exploration",
        )

        try:
            from src.config.llm_adapter import LLMResponse
            response = await self.llm.generate(prompt, temperature=0.8)
            reflection = response.text.strip()

            if reflection and self._on_promotion:
                reflection_chain = StoredChain(
                    id=f"reflection-{int(time.time())}",
                    seed_topic="daily reflection",
                    endpoint_topic="synthesis",
                    chain_summary="Daily synthesis of incubated ideas",
                    domains=[],
                    domain_crossings=0,
                    score=0.9,
                    interjection_text=reflection,
                    status="fired",
                    context_at_time=self._current_context,
                )
                await self._on_promotion(reflection_chain, 0.9)

        except Exception as e:
            print(f"   [Incubation] Daily reflection failed: {e}")
