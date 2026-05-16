"""
User Profile — Auto-generated creative identity from interaction history.

Instead of manually defining "Steven likes X and Y", this builds a living
profile from actual rated chains. After enough data accumulates, it knows:
  - Your favorite domain crossings (games → neuroscience, LEDs → biology)
  - What surprise level you prefer (wild leaps vs. elegant bridges)
  - Your humor style and engagement patterns
  - Topics you consistently rate highly

This profile gets injected into every bridge prompt so the engine speaks
with accumulated knowledge of who you are and what lights you up.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.config.llm_adapter import LLMAdapter
from src.memory.store import MemoryStore


PROFILE_GENERATION_PROMPT = """Based on the following interaction data from a creative AI companion, build a personality snapshot of the user. This will be injected into every future creative prompt.

TOP-RATED ASSOCIATIONS (things the user loved):
{top_rated}

DOMAIN PREFERENCES (which domain crossings score highest):
{domain_prefs}

RECENT INTERESTS (what they've been engaged with lately):
{recent}

STATS:
- Total interjections delivered: {total_count}
- Average user rating: {avg_rating}
- Favorite domain crossings: {fav_domains}

Write a 2-3 sentence personality profile for this user in second person ("You are Steven's..."). Focus on:
1. What KINDS of associations they love (wild/subtle/practical?)
2. Their domain interests and recurring themes
3. Any personality traits you can infer (humor, energy, curiosity style)

Example format:
"You're Steven's slightly cheeky creative buddy who knows he's fascinated by the intersection of game design and neuroscience. He loves wild, far-reaching leaps more than safe connections — the crazier the bridge, the better. He tends to get fired up about anything involving LEDs, endurance gaming, and biological systems."

Keep it warm, specific, and useful for guiding creative associations. Return just the profile text."""


@dataclass
class UserProfile:
    """The user's creative personality as learned from interactions."""
    profile_text: str = ""
    favorite_domains: list[str] = field(default_factory=list)
    preferred_surprise_level: float = 0.6
    total_rated: int = 0
    avg_rating: float = 0.0
    last_updated: float = 0.0
    update_count: int = 0


class ProfileBuilder:
    """Builds and maintains a user profile from memory data.

    The profile is rebuilt periodically (every N new ratings) so it
    evolves as the user's interests shift. Stored as a simple JSON
    file alongside the ChromaDB data.
    """

    def __init__(
        self,
        memory: MemoryStore,
        llm: LLMAdapter,
        profile_path: str = "data/memory/user_profile.json",
        rebuild_every_n_ratings: int = 10,
    ):
        self.memory = memory
        self.llm = llm
        self.profile_path = Path(profile_path)
        self.rebuild_threshold = rebuild_every_n_ratings
        self.profile = UserProfile()
        self._ratings_since_rebuild = 0

    def initialize(self) -> None:
        """Load existing profile from disk if available."""
        if self.profile_path.exists():
            try:
                data = json.loads(self.profile_path.read_text(encoding="utf-8"))
                self.profile = UserProfile(
                    profile_text=data.get("profile_text", ""),
                    favorite_domains=data.get("favorite_domains", []),
                    preferred_surprise_level=data.get("preferred_surprise_level", 0.6),
                    total_rated=data.get("total_rated", 0),
                    avg_rating=data.get("avg_rating", 0.0),
                    last_updated=data.get("last_updated", 0.0),
                    update_count=data.get("update_count", 0),
                )
                if self.profile.profile_text:
                    print(f"   [Profile] Loaded — {self.profile.total_rated} rated chains, "
                          f"rebuilt {self.profile.update_count} times")
                else:
                    print("   [Profile] No profile yet — will build after enough ratings")
            except Exception as e:
                print(f"   [Profile] Load failed: {e} — starting fresh")

    def _save(self) -> None:
        """Persist profile to disk."""
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "profile_text": self.profile.profile_text,
            "favorite_domains": self.profile.favorite_domains,
            "preferred_surprise_level": self.profile.preferred_surprise_level,
            "total_rated": self.profile.total_rated,
            "avg_rating": self.profile.avg_rating,
            "last_updated": self.profile.last_updated,
            "update_count": self.profile.update_count,
        }
        self.profile_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def on_rating(self) -> None:
        """Called when the user rates an interjection. Triggers rebuild if threshold hit."""
        self._ratings_since_rebuild += 1

    @property
    def needs_rebuild(self) -> bool:
        return self._ratings_since_rebuild >= self.rebuild_threshold

    @property
    def has_profile(self) -> bool:
        return bool(self.profile.profile_text)

    @property
    def persona_injection(self) -> str:
        """Get the profile text for injection into prompts.

        Returns empty string if no profile has been built yet.
        """
        return self.profile.profile_text

    async def rebuild_if_needed(self) -> bool:
        """Rebuild the profile if enough new ratings have accumulated."""
        if not self.needs_rebuild:
            return False
        return await self.rebuild()

    async def rebuild(self) -> bool:
        """Generate a fresh profile from all memory data."""
        if not self.memory.is_available:
            return False

        top_rated = self.memory.get_top_rated(min_rating=4, limit=15)
        if len(top_rated) < 3:
            return False

        domain_prefs = self.memory.get_domain_preferences()
        recent = self.memory.get_recent(limit=10, status="fired")

        top_text = "\n".join(
            f"  - \"{c.seed_topic}\" → \"{c.endpoint_topic}\" (rating: {c.user_rating}/5, "
            f"score: {c.score:.2f})"
            for c in top_rated[:10]
        )

        domain_text = "\n".join(
            f"  - {k}: avg rating {v:.1f}" for k, v in
            sorted(domain_prefs.items(), key=lambda x: x[1], reverse=True)[:8]
        )

        recent_text = "\n".join(
            f"  - \"{c.endpoint_topic}\" (context: {c.context_at_time[:50]})"
            for c in recent[:5]
        )

        fav_domains = sorted(domain_prefs.items(), key=lambda x: x[1], reverse=True)[:3]
        fav_text = ", ".join(k for k, _ in fav_domains) if fav_domains else "still learning"

        ratings = [c.user_rating for c in top_rated if c.user_rating]
        avg = sum(ratings) / len(ratings) if ratings else 0.0

        prompt = PROFILE_GENERATION_PROMPT.format(
            top_rated=top_text or "  (not enough data yet)",
            domain_prefs=domain_text or "  (not enough data yet)",
            recent=recent_text or "  (not enough data yet)",
            total_count=self.memory.chain_count,
            avg_rating=f"{avg:.1f}",
            fav_domains=fav_text,
        )

        try:
            response = await self.llm.generate(prompt, temperature=0.7)
            new_profile = response.text.strip().strip('"')

            if new_profile and len(new_profile) > 20:
                self.profile.profile_text = new_profile
                self.profile.favorite_domains = [k for k, _ in fav_domains]
                self.profile.total_rated = len(ratings)
                self.profile.avg_rating = avg
                self.profile.last_updated = time.time()
                self.profile.update_count += 1
                self._ratings_since_rebuild = 0
                self._save()
                print(f"   [Profile] Rebuilt! ({self.profile.update_count} updates total)")
                return True
        except Exception as e:
            print(f"   [Profile] Rebuild failed: {e}")

        return False
