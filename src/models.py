"""
Data models for Computational Serendipity.

These are the core data structures that flow through the pipeline:
Heartbeat → Context → Association Tree → Scoring → Interjection
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np


DOMAINS = [
    "Science & Technology",
    "Psychology & Behavior",
    "Art & Design",
    "History & Culture",
    "Nature & Biology",
    "Philosophy & Ethics",
    "Economics & Business",
    "Mathematics & Logic",
    "Language & Literature",
    "Sports & Games",
    "Music & Sound",
    "Food & Cooking",
    "Architecture & Space",
    "Politics & Society",
    "Health & Medicine",
]


@dataclass
class AssociationNode:
    topic: str
    domain: str
    connection_reason: str
    depth: int
    embedding: np.ndarray | None = None
    parent: AssociationNode | None = field(default=None, repr=False)
    children: list[AssociationNode] = field(default_factory=list, repr=False)
    preliminary_score: float | None = None
    grounding_status: str = "unverified"  # "grounded", "inferred", "speculative", "unverified"
    grounding_source: str = ""            # URL or brief evidence description

    def chain_to_root(self) -> list[AssociationNode]:
        """Walk up to root, return path from root to self."""
        path = []
        node: AssociationNode | None = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))

    def chain_topics(self) -> list[str]:
        return [n.topic for n in self.chain_to_root()]

    def chain_domains(self) -> list[str]:
        return [n.domain for n in self.chain_to_root()]

    def domain_crossings(self) -> int:
        domains = self.chain_domains()
        return sum(1 for i in range(1, len(domains)) if domains[i] != domains[i - 1])


@dataclass
class AssociationChain:
    seed_topic: str
    nodes: list[AssociationNode]
    total_semantic_distance: float = 0.0
    domain_crossings: int = 0
    interest_score: float | None = None
    bridge_sentence: str | None = None
    metadata: dict | None = None

    @property
    def endpoint(self) -> AssociationNode:
        return self.nodes[-1]

    @property
    def endpoint_topic(self) -> str:
        return self.endpoint.topic

    def summary(self) -> str:
        return " → ".join(n.topic for n in self.nodes)


@dataclass
class ScoringBreakdown:
    semantic_distance: float
    domain_crossings: float
    surprise: float
    bridgeability: float
    novelty: float
    total: float

    def __str__(self) -> str:
        return (
            f"  semantic_distance : {self.semantic_distance:.3f} (×0.30 = {self.semantic_distance * 0.30:.3f})\n"
            f"  domain_crossings  : {self.domain_crossings:.3f} (×0.25 = {self.domain_crossings * 0.25:.3f})\n"
            f"  surprise          : {self.surprise:.3f} (×0.20 = {self.surprise * 0.20:.3f})\n"
            f"  bridgeability     : {self.bridgeability:.3f} (×0.15 = {self.bridgeability * 0.15:.3f})\n"
            f"  novelty           : {self.novelty:.3f} (×0.10 = {self.novelty * 0.10:.3f})\n"
            f"  TOTAL             : {self.total:.3f}"
        )


@dataclass
class ChannelInput:
    """One channel's contribution to the context snapshot."""
    channel: str                    # "vision", "audio", "text"
    raw_content: str = ""           # description, transcript, or user text
    novelty: float = 1.0           # 0.0 (same as before) to 1.0 (completely new)
    base_weight: float = 0.0
    effective_weight: float = 0.0   # base_weight × novelty
    available: bool = True


@dataclass
class ContextSnapshot:
    """
    Multimodal context snapshot — what the engine perceives at this moment.
    Combines vision (webcam), audio (mic), and text (user input) channels.
    """
    timestamp: datetime = field(default_factory=datetime.now)
    heartbeat_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    # Assembled seed topic (built from all channels)
    seed_topic: str = ""

    # Individual channel data
    vision: ChannelInput | None = None
    audio: ChannelInput | None = None
    screen: ChannelInput | None = None
    text: ChannelInput | None = None

    # Composite scores
    dominant_channel: str = "text"
    overall_novelty: float = 1.0

    # Raw image bytes (for passing to vision LLM)
    image_bytes: bytes | None = field(default=None, repr=False)


@dataclass
class Interjection:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: datetime = field(default_factory=datetime.now)
    heartbeat_id: str = ""
    chain: AssociationChain | None = None
    scoring: ScoringBreakdown | None = None
    interjection_text: str = ""
    context: ContextSnapshot | None = None
    search_facts: list[str] = field(default_factory=list)
    search_sources: list[str] = field(default_factory=list)


# ── Deep Thought / Collision Models ──────────────────────────────────

@dataclass
class CollisionScore:
    """Scoring breakdown for a bisociation collision between two chains."""
    causal_depth: float = 0.0       # total hops across both chains (more = more hidden)
    seed_distance: float = 0.0      # how far apart the two original seeds were
    hiddenness: float = 0.0         # is the collision buried in the middle? (not at endpoints)
    domain_span: float = 0.0        # total unique domains crossed by both chains combined
    mechanism_specificity: float = 0.0  # does a real causal mechanism exist at the collision?
    testability: float = 0.0        # can this be turned into a testable hypothesis?
    total: float = 0.0

    def __str__(self) -> str:
        return (
            f"  causal_depth          : {self.causal_depth:.3f} (x0.20 = {self.causal_depth * 0.20:.3f})\n"
            f"  seed_distance         : {self.seed_distance:.3f} (x0.20 = {self.seed_distance * 0.20:.3f})\n"
            f"  hiddenness            : {self.hiddenness:.3f} (x0.15 = {self.hiddenness * 0.15:.3f})\n"
            f"  domain_span           : {self.domain_span:.3f} (x0.15 = {self.domain_span * 0.15:.3f})\n"
            f"  mechanism_specificity : {self.mechanism_specificity:.3f} (x0.15 = {self.mechanism_specificity * 0.15:.3f})\n"
            f"  testability           : {self.testability:.3f} (x0.15 = {self.testability * 0.15:.3f})\n"
            f"  TOTAL                 : {self.total:.3f}"
        )


@dataclass
class CollisionResult:
    """Two independent causal threads that share a hidden intermediate concept.

    This is the bisociation moment — where Koestler's "matrices of thought"
    intersect. Chain A and Chain B were generated from completely different
    seeds, but their intermediate nodes overlap in embedding space.
    """
    id: str = field(default_factory=lambda: f"col-{uuid.uuid4().hex[:12]}")
    chain_a: AssociationChain | None = None
    chain_b: AssociationChain | None = None
    collision_node_a: AssociationNode | None = None
    collision_node_b: AssociationNode | None = None
    collision_concept: str = ""           # synthesized shared concept
    collision_similarity: float = 0.0     # cosine similarity at the collision
    total_causal_distance: float = 0.0    # combined semantic distance
    total_hops: int = 0                   # combined hop count
    total_domain_crossings: int = 0       # combined domain crossings
    seed_a_label: str = ""                # e.g., "context", "memory", "inverse"
    seed_b_label: str = ""
    scoring: CollisionScore | None = None
    hypothesis: str | None = None         # generated hypothesis (filled by bridge builder)
    confidence: str = ""                  # "low", "medium", "high", "oracle"
    timestamp: datetime = field(default_factory=datetime.now)
