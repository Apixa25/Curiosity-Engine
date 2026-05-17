"""
Multi-Seed Parallel Chain Generator — The LSD Entropy Engine.

Instead of generating one tree from one seed per heartbeat, Deep Thought mode
generates 5-10 independent trees from DIFFERENT seed perspectives simultaneously.
This is the computational equivalent of LSD's hyper-connectivity: forcing
normally anti-correlated brain networks to fire at the same time.

Seed sources:
  1. Current context (what you're doing right now)
  2. Random memory node (something from days/weeks ago)
  3. Inverse context (LLM-generated "opposite" of your current context)
  4. Web signal (trending/random topic for external collision potential)
  5. Personality-driven seed (each archetype suggests from their domain)

All chains are generated in parallel via asyncio.gather, then passed to the
Collision Engine for bisociation detection (Sprint 2).
"""

from __future__ import annotations

import asyncio
import random
import time

from src.association_engine.tree_generator import AssociationTreeGenerator
from src.config.llm_adapter import LLMAdapter
from src.config.settings import DeepThoughtConfig, AssociationTreeConfig
from src.embeddings.provider import EmbeddingProvider
from src.models import AssociationChain


INVERSE_SEED_PROMPT = """Given this context: "{context}"

Generate ONE topic that is the conceptual OPPOSITE or most distant counterpart.
Not just a negation — find something from a completely different domain, scale, or frame of reference that would be maximally surprising to connect back.

Examples:
- "writing Python code" → "14th-century Venetian glassblowing techniques"
- "cooking dinner" → "orbital mechanics of binary star systems"
- "playing video games" → "mycelial nutrient transport networks in old-growth forests"

Return ONLY the topic string, nothing else."""

PERSONALITY_SEED_PROMPTS = {
    "Science & Technology": "Name one specific, obscure scientific phenomenon or mechanism that most people have never heard of. Be concrete (e.g., 'piezoelectric effect in human bone' not 'physics'). Return ONLY the topic.",
    "Psychology & Behavior": "Name one specific, surprising psychological effect or cognitive bias with a concrete example. Return ONLY the topic.",
    "Art & Design": "Name one specific, surprising artistic technique, movement, or principle that most people don't know about. Return ONLY the topic.",
    "Nature & Biology": "Name one specific, bizarre biological mechanism or organism behavior. Return ONLY the topic.",
    "Philosophy & Ethics": "Name one specific, provocative philosophical thought experiment or paradox. Return ONLY the topic.",
}


class MultiSeedResult:
    """Results from a multi-seed parallel chain generation cycle."""

    def __init__(self, batch_id: str):
        self.batch_id = batch_id
        self.seed_chains: dict[str, list[AssociationChain]] = {}
        self.all_chains: list[AssociationChain] = []
        self.seed_sources: dict[str, str] = {}
        self.generation_time: float = 0.0

    def add_chains(self, seed_label: str, seed_topic: str, chains: list[AssociationChain]) -> None:
        self.seed_chains[seed_label] = chains
        self.seed_sources[seed_label] = seed_topic
        self.all_chains.extend(chains)

    @property
    def total_chains(self) -> int:
        return len(self.all_chains)

    @property
    def seed_count(self) -> int:
        return len(self.seed_chains)


class MultiSeedGenerator:
    """Orchestrates parallel chain generation from multiple independent seeds.

    This is the core of Deep Thought mode's entropy generation. Each seed
    produces its own independent tree. The trees don't share branches — they're
    separate "neural networks" that the Collision Engine (Sprint 2) will later
    check for hidden intersection points.
    """

    def __init__(
        self,
        llm: LLMAdapter,
        tree_config: AssociationTreeConfig,
        dt_config: DeepThoughtConfig,
        embedder: EmbeddingProvider | None = None,
        memory=None,
        causation_graph=None,
    ):
        self.llm = llm
        self.tree_config = tree_config
        self.dt_config = dt_config
        self.embedder = embedder
        self.memory = memory
        self.causation_graph = causation_graph

    def _create_deep_thought_tree_gen(self) -> AssociationTreeGenerator:
        """Create a tree generator configured for Deep Thought mode."""
        return AssociationTreeGenerator(
            llm=self.llm,
            config=self.tree_config,
            embedder=self.embedder,
            deep_thought_mode=True,
            deep_thought_max_depth=self.dt_config.max_depth,
            deep_thought_keep_per_level=self.dt_config.keep_per_level,
            deep_thought_min_domain_crossings=self.dt_config.min_domain_crossings,
            deep_thought_temperature_boost=self.dt_config.llm_temperature_boost,
            deep_thought_invert_efficiency=self.dt_config.invert_efficiency,
        )

    async def generate_parallel(
        self,
        current_context: str,
        heartbeat_id: str = "",
    ) -> MultiSeedResult:
        """Generate chains from multiple seeds in parallel.

        This is the main entry point for Deep Thought mode. It:
        1. Determines seed sources (context, memory, inverse, personality)
        2. Launches independent tree generators for each seed
        3. Collects all chains into a MultiSeedResult for collision detection
        """
        batch_id = f"dt-{heartbeat_id or 'manual'}-{int(time.time())}"
        result = MultiSeedResult(batch_id)
        t0 = time.time()

        seeds = await self._gather_seeds(current_context)

        print(f"\n   🔮 DEEP THOUGHT: Launching {len(seeds)} parallel chain generators...")
        for label, topic in seeds.items():
            print(f"      {label}: \"{topic[:60]}{'...' if len(topic) > 60 else ''}\"")

        tasks = {}
        for label, seed_topic in seeds.items():
            tree_gen = self._create_deep_thought_tree_gen()
            tasks[label] = asyncio.create_task(
                self._generate_with_label(tree_gen, label, seed_topic)
            )

        completed = await asyncio.gather(*tasks.values(), return_exceptions=True)

        for (label, _), chains_or_error in zip(tasks.items(), completed):
            seed_topic = seeds[label]
            if isinstance(chains_or_error, Exception):
                print(f"      ⚠️  {label} failed: {chains_or_error}")
                result.add_chains(label, seed_topic, [])
            else:
                result.add_chains(label, seed_topic, chains_or_error)
                print(f"      ✅ {label}: {len(chains_or_error)} chains")

        result.generation_time = time.time() - t0
        print(f"\n   🔮 DEEP THOUGHT complete: {result.total_chains} total chains from {result.seed_count} seeds ({result.generation_time:.1f}s)")

        return result

    async def _generate_with_label(
        self,
        tree_gen: AssociationTreeGenerator,
        label: str,
        seed_topic: str,
    ) -> list[AssociationChain]:
        """Generate a tree for a single seed. Wraps generate_tree with error handling."""
        try:
            return await tree_gen.generate_tree(seed_topic)
        except Exception as e:
            print(f"      ⚠️  Tree generation failed for {label}: {e}")
            return []

    async def _gather_seeds(self, current_context: str) -> dict[str, str]:
        """Build the set of seeds for this cycle.

        Always includes current_context. Other seeds are added based on
        available resources (memory, graph periphery, LLM for inverse)
        up to parallel_seeds limit.

        When the causation graph is available, one seed comes from graph
        periphery (frontier territory with maximum collision potential) and
        one from high-affinity nodes (concepts the user has responded to).
        """
        seeds: dict[str, str] = {}
        target_count = self.dt_config.parallel_seeds

        seeds["context"] = current_context

        seed_tasks = []

        if target_count >= 2:
            seed_tasks.append(("inverse", self._generate_inverse_seed(current_context)))

        if target_count >= 3 and self.memory and self.memory.is_available:
            seed_tasks.append(("memory", self._get_memory_seed()))

        if target_count >= 4:
            graph_seed = self._get_graph_periphery_seed()
            if graph_seed:
                seeds["graph_frontier"] = graph_seed
            else:
                domain = random.choice(list(PERSONALITY_SEED_PROMPTS.keys()))
                seed_tasks.append((f"personality ({domain})", self._generate_personality_seed(domain)))

        if target_count >= 5:
            affinity_seed = self._get_graph_affinity_seed()
            if affinity_seed:
                seeds["graph_affinity"] = affinity_seed
            else:
                remaining_domains = [d for d in PERSONALITY_SEED_PROMPTS if f"personality ({d})" not in seeds]
                if remaining_domains:
                    domain2 = random.choice(remaining_domains)
                    seed_tasks.append((f"personality ({domain2})", self._generate_personality_seed(domain2)))

        if seed_tasks:
            labels = [label for label, _ in seed_tasks]
            coroutines = [coro for _, coro in seed_tasks]
            results = await asyncio.gather(*coroutines, return_exceptions=True)

            for label, result in zip(labels, results):
                if isinstance(result, Exception) or not result:
                    continue
                seeds[label] = result

        return seeds

    async def _generate_inverse_seed(self, context: str) -> str:
        """Use the LLM to generate a conceptually opposite seed topic."""
        try:
            prompt = INVERSE_SEED_PROMPT.format(context=context)
            response = await self.llm.generate(prompt, temperature=1.0)
            return response.text.strip().strip('"').strip("'")
        except Exception:
            return ""

    async def _get_memory_seed(self) -> str:
        """Pull a random topic from long-term memory for cross-temporal potential."""
        try:
            from src.memory.store import StoredChain
            all_fired = self.memory.get_all_fired(limit=50)
            if not all_fired:
                return ""
            chosen = random.choice(all_fired)
            return chosen.endpoint_topic
        except Exception:
            return ""

    async def _generate_personality_seed(self, domain: str) -> str:
        """Generate a domain-specific seed from a personality archetype."""
        try:
            prompt = PERSONALITY_SEED_PROMPTS.get(domain, "Name one obscure topic. Return ONLY the topic.")
            response = await self.llm.generate(prompt, temperature=1.1)
            return response.text.strip().strip('"').strip("'")
        except Exception:
            return ""

    def _get_graph_periphery_seed(self) -> str | None:
        """Get a frontier node from the causation graph for maximum collision potential.

        Periphery nodes sit at the edges of the user's known concept universe.
        Chains starting from here explore genuinely new territory — maximizing
        the chance of unexpected collisions with other chains.
        """
        if not self.causation_graph or self.causation_graph.node_count < 10:
            return None
        return self.causation_graph.get_periphery_seed()

    def _get_graph_affinity_seed(self) -> str | None:
        """Get a high-affinity node — something the user has rated highly.

        Seeds from high-affinity nodes produce chains the user is more likely
        to find interesting, based on their rating history propagated through
        the graph.
        """
        if not self.causation_graph or self.causation_graph.node_count < 10:
            return None
        return self.causation_graph.get_high_affinity_seed()
