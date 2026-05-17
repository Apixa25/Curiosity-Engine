"""
Interest Scorer — Evaluates association chains for interestingness.

Five metrics, weighted per spec:
  semantic_distance  × 0.30  — reward boldness (REAL cosine distance with embeddings)
  domain_crossings   × 0.25  — reward crossing fields
  surprise           × 0.20  — reward the unexpected
  bridgeability      × 0.15  — can a story be told?
  novelty            × 0.10  — haven't said this before (embedding similarity with history)

When embeddings are available, semantic_distance uses real cosine distance
between the seed and endpoint in embedding space. The distance is also
weighted by an "efficiency ratio" per Kenett et al. (2014-2018): chains
that reach far in fewer hops score higher — mimicking the flat associative
networks of highly creative people.
"""

from __future__ import annotations

import math

from src.config.llm_adapter import LLMAdapter
from src.config.settings import ScoringConfig, DeepThoughtScoringWeights
from src.embeddings.provider import EmbeddingProvider, cosine_distance, cosine_similarity
from src.models import AssociationChain, ScoringBreakdown, ContextSnapshot, CollisionResult, CollisionScore

import numpy as np


MECHANISM_PROMPT = """Two independent chains of thought collided at a hidden intersection:

Chain A: {chain_a_summary}
Chain B: {chain_b_summary}
Collision point: "{collision_concept}"

On a scale of 0.0 to 1.0, how SPECIFIC and REAL is the causal mechanism at this collision point?

0.0 = no real mechanism, just vague thematic similarity
0.2 = weak analogy, no causal link
0.4 = plausible but unverified mechanism
0.6 = real mechanism that exists but is not widely known
0.8 = well-documented causal pathway connecting these domains
1.0 = published, peer-reviewed causal mechanism

Be critical. Most collisions are 0.2-0.5.
Return ONLY a single number, nothing else."""

TESTABILITY_PROMPT = """A hidden causal chain connects two completely different domains:

Starting points: "{seed_a}" and "{seed_b}"
Collision concept: "{collision_concept}"
Combined chain crosses {domain_crossings} domain boundaries in {total_hops} hops.

On a scale of 0.0 to 1.0, how TESTABLE is the hypothesis implied by this connection?

0.0 = pure philosophical speculation, no way to test
0.2 = vaguely interesting but no clear experiment
0.4 = could design an experiment but it would be expensive/complex
0.6 = testable with reasonable effort (literature review, small experiment, data analysis)
0.8 = easily testable with a simple experiment or observable prediction
1.0 = immediately verifiable — you could check this right now

Be critical. Most connections are 0.2-0.5.
Return ONLY a single number, nothing else."""

SURPRISE_PROMPT = """On a scale of 0.0 to 1.0, how SURPRISING is it to connect "{seed}" to "{endpoint}"?

Be a tough critic. Most connections are only moderately surprising.

0.0 = completely obvious, anyone would connect these
0.2 = mildly interesting but predictable
0.4 = somewhat surprising, a decent lateral jump
0.6 = genuinely surprising, most people wouldn't see this connection
0.8 = very surprising, a creative cross-domain leap
1.0 = extraordinary, a once-in-a-lifetime "aha!" connection

The chain was: {chain_summary}

Be honest and critical. Most associations deserve 0.3-0.6.
Return ONLY a single number, nothing else."""

BRIDGEABILITY_PROMPT = """A creative association chain went from "{seed}" to "{endpoint}" through:
{chain_summary}

The user is currently focused on: "{context}"

Could you tell a compelling 1-sentence story connecting "{endpoint}" back to "{context}" in a way that would make someone say "huh, that's interesting"?

Rate honestly on 0.0 to 1.0:
0.0 = no meaningful connection, pure noise
0.3 = very tenuous, really stretching
0.5 = there's a connection but it requires explanation
0.7 = solid connection, would genuinely interest someone
1.0 = brilliant, illuminating connection

Most connections are 0.3-0.6. Be critical.
Return ONLY a single number, nothing else."""


class InterestScorer:
    def __init__(
        self,
        llm: LLMAdapter,
        config: ScoringConfig | None = None,
        embedder: EmbeddingProvider | None = None,
    ):
        self.llm = llm
        self.cfg = config or ScoringConfig()
        self.embedder = embedder
        self._past_embeddings: list[np.ndarray] = []

    async def score_chain(
        self,
        chain: AssociationChain,
        context: ContextSnapshot,
        past_interjection_topics: list[str] | None = None,
    ) -> ScoringBreakdown:
        """
        Score a single association chain on all five metrics.
        Uses real embeddings when available for semantic_distance and novelty.
        """
        w = self.cfg.weights

        sd = self._compute_semantic_distance(chain)
        dc = self._compute_domain_crossing_score(chain)
        surprise = await self._compute_surprise(chain)
        bridge = await self._compute_bridgeability(chain, context)
        novelty = self._compute_novelty(chain, past_interjection_topics or [])

        total = (
            sd * w.semantic_distance
            + dc * w.domain_crossings
            + surprise * w.surprise
            + bridge * w.bridgeability
            + novelty * w.novelty
        )

        return ScoringBreakdown(
            semantic_distance=sd,
            domain_crossings=dc,
            surprise=surprise,
            bridgeability=bridge,
            novelty=novelty,
            total=total,
        )

    def _compute_semantic_distance(self, chain: AssociationChain) -> float:
        """
        Compute semantic distance between seed and endpoint.

        With embeddings: uses real cosine distance, boosted by an efficiency
        ratio (distance / hops). This rewards chains that reach far in fewer
        hops — the "flat hierarchy" of creative minds per Kenett et al.
        A chain that gets from "silver coins" to "cancer cure" in 4 hops
        scores higher than one that takes 7 hops to get the same distance.

        Without embeddings: falls back to the log-curve heuristic based on
        depth and domain crossings (original behavior).
        """
        root = chain.nodes[0] if chain.nodes else None
        endpoint = chain.nodes[-1] if chain.nodes else None

        if (root and endpoint
                and root.embedding is not None
                and endpoint.embedding is not None):
            raw_distance = cosine_distance(root.embedding, endpoint.embedding)
            hops = max(len(chain.nodes) - 1, 1)
            efficiency = raw_distance / hops
            efficiency_bonus = min(1.0, efficiency * 3.0)
            score = (raw_distance * 0.7) + (efficiency_bonus * 0.3)
            return min(1.0, score)

        depth = len(chain.nodes) - 1
        crossings = chain.domain_crossings
        depth_score = min(1.0, math.log(1 + depth) / math.log(1 + 8))
        crossing_score = min(1.0, math.log(1 + crossings) / math.log(1 + 5))
        return (depth_score * 0.4) + (crossing_score * 0.6)

    def _compute_domain_crossing_score(self, chain: AssociationChain) -> float:
        """Normalized domain crossing score per spec."""
        crossings = chain.domain_crossings
        scoring_map = {0: 0.0, 1: 0.4, 2: 0.7}
        return scoring_map.get(crossings, 1.0)

    async def _compute_surprise(self, chain: AssociationChain) -> float:
        """Ask the LLM how surprising this connection is."""
        prompt = SURPRISE_PROMPT.format(
            seed=chain.seed_topic,
            endpoint=chain.endpoint_topic,
            chain_summary=chain.summary(),
        )
        try:
            return await self.llm.generate_float(prompt)
        except (ValueError, Exception) as e:
            print(f"   ⚠️  Surprise scoring failed: {e}, defaulting to 0.5")
            return 0.5

    async def _compute_bridgeability(self, chain: AssociationChain, context: ContextSnapshot) -> float:
        """Ask the LLM if a compelling bridge can be told."""
        prompt = BRIDGEABILITY_PROMPT.format(
            seed=chain.seed_topic,
            endpoint=chain.endpoint_topic,
            chain_summary=chain.summary(),
            context=context.seed_topic or chain.seed_topic,
        )
        try:
            return await self.llm.generate_float(prompt)
        except (ValueError, Exception) as e:
            print(f"   ⚠️  Bridgeability scoring failed: {e}, defaulting to 0.5")
            return 0.5

    def _compute_novelty(self, chain: AssociationChain, past_topics: list[str]) -> float:
        """
        Check how novel this endpoint is compared to past interjections.

        With embeddings: computes cosine similarity against all past endpoint
        embeddings. If any past topic is >0.85 similar, it's a near-duplicate.
        Continuous scoring means "kinda similar" topics still get partial credit.

        Without embeddings: falls back to substring matching (original behavior).
        """
        endpoint = chain.nodes[-1] if chain.nodes else None

        if endpoint and endpoint.embedding is not None and self._past_embeddings:
            max_sim = 0.0
            for past_emb in self._past_embeddings:
                sim = cosine_similarity(endpoint.embedding, past_emb)
                max_sim = max(max_sim, sim)
            return max(0.0, 1.0 - max_sim)

        if not past_topics:
            return 1.0

        endpoint_lower = chain.endpoint_topic.lower()
        for past in past_topics:
            if past.lower() in endpoint_lower or endpoint_lower in past.lower():
                return 0.1
        return 1.0

    async def score_collision(
        self,
        collision: CollisionResult,
        weights: DeepThoughtScoringWeights | None = None,
    ) -> CollisionScore:
        """Score a bisociation collision on the Deep Thought metrics.

        Unlike normal chain scoring (5 metrics weighted for conversational
        interjections), collision scoring rewards depth, hiddenness, and
        mechanism specificity — the qualities that make a collision feel
        like genuine discovery rather than noise.
        """
        w = weights or DeepThoughtScoringWeights()

        causal_depth = self._score_causal_depth(collision)
        seed_distance = self._score_seed_distance(collision)
        hiddenness = self._score_hiddenness(collision)
        domain_span = self._score_domain_span(collision)
        mechanism = await self._score_mechanism(collision)
        testability = await self._score_testability(collision)

        total = (
            causal_depth * w.causal_depth
            + seed_distance * w.seed_distance
            + hiddenness * w.hiddenness
            + domain_span * w.domain_span
            + mechanism * w.mechanism_specificity
            + testability * w.testability
        )

        return CollisionScore(
            causal_depth=causal_depth,
            seed_distance=seed_distance,
            hiddenness=hiddenness,
            domain_span=domain_span,
            mechanism_specificity=mechanism,
            testability=testability,
            total=total,
        )

    def _score_causal_depth(self, collision: CollisionResult) -> float:
        """More total hops = more hidden causation. Normalized with log scaling."""
        hops = collision.total_hops
        return min(1.0, math.log(1 + hops) / math.log(1 + 30))

    def _score_seed_distance(self, collision: CollisionResult) -> float:
        """How far apart are the two seeds in embedding space?
        Farther apart seeds colliding = more surprising."""
        if collision.chain_a and collision.chain_b:
            root_a = collision.chain_a.nodes[0] if collision.chain_a.nodes else None
            root_b = collision.chain_b.nodes[0] if collision.chain_b.nodes else None
            if (root_a and root_b
                    and root_a.embedding is not None
                    and root_b.embedding is not None):
                dist = cosine_distance(root_a.embedding, root_b.embedding)
                return min(1.0, dist * 1.5)
        return 0.5

    def _score_hiddenness(self, collision: CollisionResult) -> float:
        """Is the collision buried in the MIDDLE of the chains?
        Collisions at endpoints are obvious. Collisions at depth 4+ in
        a 10-hop chain are invisible to anyone seeing just the surface."""
        if not collision.collision_node_a or not collision.collision_node_b:
            return 0.5

        depth_a = collision.collision_node_a.depth
        depth_b = collision.collision_node_b.depth
        chain_len_a = len(collision.chain_a.nodes) if collision.chain_a else 1
        chain_len_b = len(collision.chain_b.nodes) if collision.chain_b else 1

        ratio_a = depth_a / max(chain_len_a - 1, 1)
        ratio_b = depth_b / max(chain_len_b - 1, 1)

        midness_a = 1.0 - abs(ratio_a - 0.5) * 2.0
        midness_b = 1.0 - abs(ratio_b - 0.5) * 2.0

        return (midness_a + midness_b) / 2.0

    def _score_domain_span(self, collision: CollisionResult) -> float:
        """Total unique domains crossed by both chains. More = more creative."""
        crossings = collision.total_domain_crossings
        return min(1.0, math.log(1 + crossings) / math.log(1 + 10))

    async def _score_mechanism(self, collision: CollisionResult) -> float:
        """Ask LLM if there's a real causal mechanism at the collision."""
        try:
            prompt = MECHANISM_PROMPT.format(
                chain_a_summary=collision.chain_a.summary() if collision.chain_a else "",
                chain_b_summary=collision.chain_b.summary() if collision.chain_b else "",
                collision_concept=collision.collision_concept,
            )
            return await self.llm.generate_float(prompt)
        except Exception:
            return 0.4

    async def _score_testability(self, collision: CollisionResult) -> float:
        """Ask LLM how testable the implied hypothesis is."""
        try:
            prompt = TESTABILITY_PROMPT.format(
                seed_a=collision.seed_a_label,
                seed_b=collision.seed_b_label,
                collision_concept=collision.collision_concept,
                domain_crossings=collision.total_domain_crossings,
                total_hops=collision.total_hops,
            )
            return await self.llm.generate_float(prompt)
        except Exception:
            return 0.4

    def record_interjection(self, chain: AssociationChain) -> None:
        """Record a chain's endpoint embedding for future novelty checks."""
        endpoint = chain.nodes[-1] if chain.nodes else None
        if endpoint and endpoint.embedding is not None:
            self._past_embeddings.append(endpoint.embedding)

    async def rank_chains(
        self,
        chains: list[AssociationChain],
        context: ContextSnapshot,
        past_interjection_topics: list[str] | None = None,
        max_to_score: int = 5,
    ) -> list[tuple[AssociationChain, ScoringBreakdown]]:
        """
        Pre-filter chains using cheap metrics (no LLM calls), then only
        full-score the top candidates. This keeps LLM calls bounded.
        With embeddings, pre-filtering uses real semantic distance.
        """
        past = past_interjection_topics or []

        pre_scored = []
        for chain in chains:
            sd = self._compute_semantic_distance(chain)
            dc = self._compute_domain_crossing_score(chain)
            novelty = self._compute_novelty(chain, past)
            cheap_score = sd * 0.4 + dc * 0.35 + novelty * 0.25
            pre_scored.append((chain, cheap_score))

        pre_scored.sort(key=lambda x: x[1], reverse=True)
        top_chains = [c for c, _ in pre_scored[:max_to_score]]

        print(f"   📋 Pre-filtered {len(chains)} chains → top {len(top_chains)} for full scoring")

        results = []
        for chain in top_chains:
            score = await self.score_chain(chain, context, past)
            chain.interest_score = score.total
            results.append((chain, score))

        results.sort(key=lambda x: x[1].total, reverse=True)
        return results
