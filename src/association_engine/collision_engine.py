"""
Collision Engine — Bisociation Detection Between Independent Causal Chains.

This is the computational heart of Deep Thought mode. It takes chains
generated from DIFFERENT seeds and finds "collision points" — places
where two independent causal threads pass through the same conceptual
region in embedding space.

This is the deterministic equivalent of LSD's hyper-connectivity:
normally anti-correlated brain networks (separate chains from different
seeds) firing simultaneously and discovering they share a hidden
intermediate concept. Koestler called this "bisociation" — the collision
of two independent matrices of thought.

The collision point is invisible to any single chain. It only appears
when you compare ACROSS chains. That's the magic.
"""

from __future__ import annotations

import asyncio
from itertools import combinations

import numpy as np

from src.config.llm_adapter import LLMAdapter
from src.embeddings.provider import cosine_similarity, cosine_distance
from src.models import (
    AssociationChain,
    AssociationNode,
    CollisionResult,
)


COLLISION_CONCEPT_PROMPT = """Two completely independent chains of thought collided at a hidden intersection point.

Chain A started from: "{seed_a}"
Chain A's node near the collision: "{node_a_topic}" (domain: {node_a_domain})
Chain A's connection reason: "{node_a_reason}"

Chain B started from: "{seed_b}"
Chain B's node near the collision: "{node_b_topic}" (domain: {node_b_domain})
Chain B's connection reason: "{node_b_reason}"

These two nodes are semantically similar despite coming from completely different starting points.

In ONE concise phrase (5-10 words), what is the SHARED CONCEPT that connects these two nodes?
Name the specific mechanism, phenomenon, or principle they have in common.

Return ONLY the concept phrase, nothing else."""


class CollisionEngine:
    """Detects bisociation points between independent association chains.

    Takes all chains from a multi-seed generation cycle and finds pairs
    of intermediate nodes (from DIFFERENT seed sources) that are close
    in embedding space. Each such pair is a potential "butterfly-effect"
    discovery — a hidden link between causally unrelated domains.
    """

    def __init__(
        self,
        llm: LLMAdapter,
        collision_threshold: float = 0.82,
    ):
        self.llm = llm
        self.threshold = collision_threshold

    async def detect_collisions(
        self,
        seed_chains: dict[str, list[AssociationChain]],
        seed_sources: dict[str, str],
        max_collisions: int = 5,
    ) -> list[CollisionResult]:
        """Find bisociation points between chains from different seeds.

        This is the core algorithm:
        1. Collect all intermediate nodes (not endpoints) from all chains
        2. Group them by seed label so we only compare ACROSS seeds
        3. For each cross-seed node pair, compute cosine similarity
        4. Pairs above threshold are collision candidates
        5. Synthesize the shared concept via LLM
        6. Build CollisionResult objects

        We skip endpoint nodes because collisions at chain ENDS are obvious.
        The magic is when the collision is BURIED in the middle — hidden
        from anyone who only sees the chain's starting and ending points.
        """
        node_pool = self._collect_intermediate_nodes(seed_chains)

        if len(node_pool) < 2:
            return []

        seed_labels = list(node_pool.keys())
        if len(seed_labels) < 2:
            return []

        raw_collisions = self._find_embedding_collisions(node_pool, seed_labels)

        if not raw_collisions:
            return []

        raw_collisions.sort(key=lambda x: x["similarity"], reverse=True)
        top_candidates = raw_collisions[:max_collisions]

        results = []
        for candidate in top_candidates:
            collision = await self._build_collision_result(
                candidate, seed_chains, seed_sources
            )
            if collision:
                results.append(collision)

        return results

    def _collect_intermediate_nodes(
        self,
        seed_chains: dict[str, list[AssociationChain]],
    ) -> dict[str, list[tuple[AssociationNode, AssociationChain]]]:
        """Collect intermediate nodes (excluding root and endpoint) from each seed group.

        Returns {seed_label: [(node, parent_chain), ...]}
        Only includes nodes that have embeddings.
        """
        pool: dict[str, list[tuple[AssociationNode, AssociationChain]]] = {}

        for seed_label, chains in seed_chains.items():
            nodes = []
            for chain in chains:
                if len(chain.nodes) < 3:
                    continue
                for node in chain.nodes[1:-1]:
                    if node.embedding is not None:
                        nodes.append((node, chain))
            if nodes:
                pool[seed_label] = nodes

        return pool

    def _find_embedding_collisions(
        self,
        node_pool: dict[str, list[tuple[AssociationNode, AssociationChain]]],
        seed_labels: list[str],
    ) -> list[dict]:
        """Brute-force pairwise cosine similarity across nodes from different seeds.

        For small-to-medium node counts (typical: 50-200 nodes across 5 seeds),
        this is fast enough. For larger sets, we'd switch to approximate
        nearest-neighbor search (FAISS/Annoy).
        """
        collisions = []

        for label_a, label_b in combinations(seed_labels, 2):
            nodes_a = node_pool.get(label_a, [])
            nodes_b = node_pool.get(label_b, [])

            for node_a, chain_a in nodes_a:
                for node_b, chain_b in nodes_b:
                    if node_a.embedding is None or node_b.embedding is None:
                        continue

                    sim = cosine_similarity(node_a.embedding, node_b.embedding)

                    if sim >= self.threshold:
                        collisions.append({
                            "node_a": node_a,
                            "chain_a": chain_a,
                            "label_a": label_a,
                            "node_b": node_b,
                            "chain_b": chain_b,
                            "label_b": label_b,
                            "similarity": sim,
                        })

        return collisions

    async def _build_collision_result(
        self,
        candidate: dict,
        seed_chains: dict[str, list[AssociationChain]],
        seed_sources: dict[str, str],
    ) -> CollisionResult | None:
        """Build a full CollisionResult from a raw collision candidate."""
        node_a: AssociationNode = candidate["node_a"]
        node_b: AssociationNode = candidate["node_b"]
        chain_a: AssociationChain = candidate["chain_a"]
        chain_b: AssociationChain = candidate["chain_b"]
        label_a: str = candidate["label_a"]
        label_b: str = candidate["label_b"]
        similarity: float = candidate["similarity"]

        collision_concept = await self._synthesize_concept(
            node_a, node_b,
            seed_sources.get(label_a, ""),
            seed_sources.get(label_b, ""),
        )

        total_hops = (len(chain_a.nodes) - 1) + (len(chain_b.nodes) - 1)
        total_crossings = chain_a.domain_crossings + chain_b.domain_crossings

        total_distance = chain_a.total_semantic_distance + chain_b.total_semantic_distance

        return CollisionResult(
            chain_a=chain_a,
            chain_b=chain_b,
            collision_node_a=node_a,
            collision_node_b=node_b,
            collision_concept=collision_concept,
            collision_similarity=similarity,
            total_causal_distance=total_distance,
            total_hops=total_hops,
            total_domain_crossings=total_crossings,
            seed_a_label=label_a,
            seed_b_label=label_b,
        )

    async def _synthesize_concept(
        self,
        node_a: AssociationNode,
        node_b: AssociationNode,
        seed_a: str,
        seed_b: str,
    ) -> str:
        """Use LLM to name the shared concept at the collision point."""
        try:
            prompt = COLLISION_CONCEPT_PROMPT.format(
                seed_a=seed_a,
                node_a_topic=node_a.topic,
                node_a_domain=node_a.domain,
                node_a_reason=node_a.connection_reason,
                seed_b=seed_b,
                node_b_topic=node_b.topic,
                node_b_domain=node_b.domain,
                node_b_reason=node_b.connection_reason,
            )
            response = await self.llm.generate(prompt, temperature=0.7)
            return response.text.strip().strip('"').strip("'")
        except Exception:
            return f"{node_a.topic} / {node_b.topic}"
