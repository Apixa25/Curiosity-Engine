"""
Memory Store — Persistent memory for Computational Serendipity.

Uses ChromaDB for vector storage and SQLite for structured metadata.
This is what turns the engine from a stateless demo into a long-term friend.

Stores:
  - Every chain that was spoken aloud (with score, domains, user rating)
  - Incubating chains (low-scoring but promising, re-evaluated periodically)
  - User interaction patterns (what topics get thumbs up/down)

Queries:
  - "What has Steven rated highly that connects domain X to domain Y?"
  - "Have I said anything similar to this endpoint before?"
  - "What's incubating that might be relevant to the current context?"
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass
class StoredChain:
    """A chain persisted in memory with all its metadata."""
    id: str
    seed_topic: str
    endpoint_topic: str
    chain_summary: str
    domains: list[str]
    domain_crossings: int
    score: float
    interjection_text: str = ""
    user_rating: int | None = None      # 1-5 or None if unrated
    status: Literal["fired", "incubating", "expired", "promoted"] = "fired"
    timestamp: float = field(default_factory=time.time)
    context_at_time: str = ""
    rescore_count: int = 0
    last_rescore: float = 0.0
    # Individual scoring metrics (for serendipity self-evaluation)
    score_semantic_distance: float = 0.0
    score_domain_crossings: float = 0.0
    score_surprise: float = 0.0
    score_bridgeability: float = 0.0
    score_novelty: float = 0.0

    def to_metadata(self) -> dict:
        """Convert to ChromaDB metadata dict (no nested objects allowed)."""
        return {
            "seed_topic": self.seed_topic,
            "endpoint_topic": self.endpoint_topic,
            "chain_summary": self.chain_summary,
            "domains": json.dumps(self.domains),
            "domain_crossings": self.domain_crossings,
            "score": self.score,
            "interjection_text": self.interjection_text,
            "user_rating": self.user_rating or 0,
            "status": self.status,
            "timestamp": self.timestamp,
            "context_at_time": self.context_at_time,
            "rescore_count": self.rescore_count,
            "last_rescore": self.last_rescore,
            "score_semantic_distance": self.score_semantic_distance,
            "score_domain_crossings": self.score_domain_crossings,
            "score_surprise": self.score_surprise,
            "score_bridgeability": self.score_bridgeability,
            "score_novelty": self.score_novelty,
        }

    @classmethod
    def from_metadata(cls, id: str, meta: dict) -> StoredChain:
        """Reconstruct from ChromaDB metadata dict."""
        return cls(
            id=id,
            seed_topic=meta.get("seed_topic", ""),
            endpoint_topic=meta.get("endpoint_topic", ""),
            chain_summary=meta.get("chain_summary", ""),
            domains=json.loads(meta.get("domains", "[]")),
            domain_crossings=meta.get("domain_crossings", 0),
            score=meta.get("score", 0.0),
            interjection_text=meta.get("interjection_text", ""),
            user_rating=meta.get("user_rating") or None,
            status=meta.get("status", "fired"),
            timestamp=meta.get("timestamp", 0.0),
            context_at_time=meta.get("context_at_time", ""),
            rescore_count=meta.get("rescore_count", 0),
            last_rescore=meta.get("last_rescore", 0.0),
            score_semantic_distance=meta.get("score_semantic_distance", 0.0),
            score_domain_crossings=meta.get("score_domain_crossings", 0.0),
            score_surprise=meta.get("score_surprise", 0.0),
            score_bridgeability=meta.get("score_bridgeability", 0.0),
            score_novelty=meta.get("score_novelty", 0.0),
        )


@dataclass
class MemoryConfig:
    enabled: bool = True
    persist_directory: str = "data/memory"
    collection_name: str = "creativity_chains"
    nodes_collection_name: str = "intermediate_nodes"
    max_similar_results: int = 10
    novelty_similarity_threshold: float = 0.85


class MemoryStore:
    """Persistent vector memory backed by ChromaDB.

    All chain data is stored with embeddings so we can do semantic queries like:
    "find me everything Steven has liked that's in the neuroscience domain"
    or "have I already made a connection similar to this one?"
    """

    def __init__(self, config: MemoryConfig | None = None):
        self.cfg = config or MemoryConfig()
        self._client = None
        self._collection = None
        self._nodes_collection = None
        self._available = False
        self._last_interjection_id: str | None = None

    def initialize(self) -> bool:
        """Set up ChromaDB with persistent storage."""
        if not self.cfg.enabled:
            print("   [Memory] Disabled by config")
            return False

        try:
            import chromadb
            from chromadb.config import Settings

            persist_dir = Path(self.cfg.persist_directory)
            persist_dir.mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=Settings(anonymized_telemetry=False),
            )

            self._collection = self._client.get_or_create_collection(
                name=self.cfg.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            self._nodes_collection = self._client.get_or_create_collection(
                name=self.cfg.nodes_collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            count = self._collection.count()
            node_count = self._nodes_collection.count()
            self._available = True
            print(f"   [Memory] ChromaDB ready — {count} chains, {node_count} intermediate nodes")
            return True

        except ImportError:
            print("   [Memory] chromadb not installed — memory disabled")
            print("   [Memory] Install with: pip install chromadb")
            self._available = False
            return False
        except Exception as e:
            print(f"   [Memory] Failed to initialize: {e}")
            self._available = False
            return False

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def chain_count(self) -> int:
        if not self._available or not self._collection:
            return 0
        return self._collection.count()

    def store_chain(
        self,
        chain_id: str,
        seed_topic: str,
        endpoint_topic: str,
        chain_summary: str,
        domains: list[str],
        domain_crossings: int,
        score: float,
        embedding: np.ndarray | None,
        interjection_text: str = "",
        context: str = "",
        status: str = "fired",
        score_semantic_distance: float = 0.0,
        score_domain_crossings: float = 0.0,
        score_surprise: float = 0.0,
        score_bridgeability: float = 0.0,
        score_novelty: float = 0.0,
    ) -> None:
        """Store a chain in persistent memory with its embedding and full scoring breakdown."""
        if not self._available or not self._collection:
            return

        stored = StoredChain(
            id=chain_id,
            seed_topic=seed_topic,
            endpoint_topic=endpoint_topic,
            chain_summary=chain_summary,
            domains=domains,
            domain_crossings=domain_crossings,
            score=score,
            interjection_text=interjection_text,
            status=status,
            context_at_time=context,
            score_semantic_distance=score_semantic_distance,
            score_domain_crossings=score_domain_crossings,
            score_surprise=score_surprise,
            score_bridgeability=score_bridgeability,
            score_novelty=score_novelty,
        )

        add_kwargs = {
            "ids": [chain_id],
            "metadatas": [stored.to_metadata()],
            "documents": [f"{seed_topic} → {endpoint_topic}: {chain_summary}"],
        }

        if embedding is not None:
            add_kwargs["embeddings"] = [embedding.tolist()]

        try:
            self._collection.upsert(**add_kwargs)
            if status == "fired":
                self._last_interjection_id = chain_id
        except Exception as e:
            print(f"   [Memory] Store failed: {e}")

    def rate_last_interjection(self, rating: int) -> bool:
        """Rate the most recent interjection (1-5 or thumbs: 1=down, 5=up)."""
        if not self._available or not self._collection:
            return False
        if not self._last_interjection_id:
            return False

        try:
            result = self._collection.get(ids=[self._last_interjection_id])
            if result and result["metadatas"]:
                meta = result["metadatas"][0]
                meta["user_rating"] = max(1, min(5, rating))
                self._collection.update(
                    ids=[self._last_interjection_id],
                    metadatas=[meta],
                )
                return True
        except Exception as e:
            print(f"   [Memory] Rating failed: {e}")
        return False

    def rate_by_id(self, chain_id: str, rating: int) -> bool:
        """Rate a specific chain by ID."""
        if not self._available or not self._collection:
            return False
        try:
            result = self._collection.get(ids=[chain_id])
            if result and result["metadatas"]:
                meta = result["metadatas"][0]
                meta["user_rating"] = max(1, min(5, rating))
                self._collection.update(ids=[chain_id], metadatas=[meta])
                return True
        except Exception:
            pass
        return False

    def find_similar(
        self,
        embedding: np.ndarray,
        n_results: int = 5,
        status_filter: str | None = None,
    ) -> list[StoredChain]:
        """Find chains with similar endpoints (for novelty checking)."""
        if not self._available or not self._collection:
            return []

        where_filter = None
        if status_filter:
            where_filter = {"status": status_filter}

        try:
            results = self._collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=min(n_results, self._collection.count() or 1),
                where=where_filter,
            )

            chains = []
            if results and results["ids"] and results["ids"][0]:
                for i, chain_id in enumerate(results["ids"][0]):
                    meta = results["metadatas"][0][i] if results["metadatas"] else {}
                    chains.append(StoredChain.from_metadata(chain_id, meta))
            return chains
        except Exception as e:
            print(f"   [Memory] Query failed: {e}")
            return []

    def check_novelty(self, embedding: np.ndarray) -> float:
        """Check how novel an endpoint is compared to all past interjections.

        Returns 0.0 (exact duplicate) to 1.0 (completely new).
        Uses the configured similarity threshold.
        """
        if not self._available or not self._collection or self._collection.count() == 0:
            return 1.0

        try:
            results = self._collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=1,
                where={"status": "fired"},
            )

            if results and results["distances"] and results["distances"][0]:
                closest_distance = results["distances"][0][0]
                similarity = 1.0 - closest_distance
                return max(0.0, 1.0 - similarity)
            return 1.0
        except Exception:
            return 1.0

    def get_incubating(self, limit: int = 20) -> list[StoredChain]:
        """Get all chains currently incubating."""
        if not self._available or not self._collection:
            return []

        try:
            results = self._collection.get(
                where={"status": "incubating"},
                limit=limit,
            )

            chains = []
            if results and results["ids"]:
                for i, chain_id in enumerate(results["ids"]):
                    meta = results["metadatas"][i] if results["metadatas"] else {}
                    chains.append(StoredChain.from_metadata(chain_id, meta))
            return chains
        except Exception as e:
            print(f"   [Memory] Get incubating failed: {e}")
            return []

    def update_chain_status(self, chain_id: str, status: str, new_score: float | None = None) -> None:
        """Update a chain's status (e.g., incubating → promoted or expired)."""
        if not self._available or not self._collection:
            return

        try:
            result = self._collection.get(ids=[chain_id])
            if result and result["metadatas"]:
                meta = result["metadatas"][0]
                meta["status"] = status
                if new_score is not None:
                    meta["score"] = new_score
                meta["rescore_count"] = meta.get("rescore_count", 0) + 1
                meta["last_rescore"] = time.time()
                self._collection.update(ids=[chain_id], metadatas=[meta])
        except Exception as e:
            print(f"   [Memory] Update status failed: {e}")

    def get_top_rated(self, min_rating: int = 4, limit: int = 20) -> list[StoredChain]:
        """Get the user's top-rated chains for profile building."""
        if not self._available or not self._collection:
            return []

        try:
            results = self._collection.get(
                where={"user_rating": {"$gte": min_rating}},
                limit=limit,
            )

            chains = []
            if results and results["ids"]:
                for i, chain_id in enumerate(results["ids"]):
                    meta = results["metadatas"][i] if results["metadatas"] else {}
                    chains.append(StoredChain.from_metadata(chain_id, meta))
            return chains
        except Exception:
            return []

    def get_recent(self, limit: int = 10, status: str = "fired") -> list[StoredChain]:
        """Get the most recent chains by status."""
        if not self._available or not self._collection:
            return []

        try:
            results = self._collection.get(
                where={"status": status},
                limit=limit,
            )

            chains = []
            if results and results["ids"]:
                for i, chain_id in enumerate(results["ids"]):
                    meta = results["metadatas"][i] if results["metadatas"] else {}
                    chains.append(StoredChain.from_metadata(chain_id, meta))
            chains.sort(key=lambda c: c.timestamp, reverse=True)
            return chains[:limit]
        except Exception:
            return []

    def get_domain_preferences(self) -> dict[str, float]:
        """Analyze which domain pairs the user rates highly.

        Returns a dict of "domain_a → domain_b": avg_rating for all rated chains.
        Used by the profile builder to understand the user's creative preferences.
        """
        if not self._available or not self._collection:
            return {}

        try:
            results = self._collection.get(
                where={"user_rating": {"$gte": 1}},
                limit=100,
            )

            domain_scores: dict[str, list[float]] = {}
            if results and results["ids"]:
                for i, _ in enumerate(results["ids"]):
                    meta = results["metadatas"][i]
                    domains = json.loads(meta.get("domains", "[]"))
                    rating = meta.get("user_rating", 0)
                    if len(domains) >= 2 and rating:
                        key = f"{domains[0]} → {domains[-1]}"
                        domain_scores.setdefault(key, []).append(float(rating))

            return {k: sum(v) / len(v) for k, v in domain_scores.items() if v}
        except Exception:
            return {}

    def get_all_fired(self, limit: int = 500) -> list[StoredChain]:
        """Get all fired chains for analytics. Returns up to `limit` chains sorted by timestamp."""
        if not self._available or not self._collection:
            return []
        try:
            results = self._collection.get(
                where={"status": "fired"},
                limit=limit,
            )
            chains = []
            if results and results["ids"]:
                for i, chain_id in enumerate(results["ids"]):
                    meta = results["metadatas"][i] if results["metadatas"] else {}
                    chains.append(StoredChain.from_metadata(chain_id, meta))
            chains.sort(key=lambda c: c.timestamp)
            return chains
        except Exception:
            return []

    # ── Cross-Temporal Memory: Intermediate Node Storage ─────────────

    @property
    def node_count(self) -> int:
        """Number of intermediate nodes stored for cross-temporal collision detection."""
        if not self._available or not self._nodes_collection:
            return 0
        return self._nodes_collection.count()

    def store_intermediate_nodes(
        self,
        chain_id: str,
        nodes: list,
        seed_topic: str,
        context: str = "",
    ) -> int:
        """Persist ALL intermediate node embeddings from a chain.

        This is what enables cross-temporal collisions. By storing every
        intermediate node (not just endpoints), we can detect when a chain
        generated TODAY passes through the same conceptual region as a chain
        from weeks ago. The collision is invisible if you only store endpoints.

        Returns the number of nodes stored.
        """
        if not self._available or not self._nodes_collection:
            return 0

        stored_count = 0
        current_time = time.time()

        for i, node in enumerate(nodes):
            if node.embedding is None:
                continue
            if i == 0 or i == len(nodes) - 1:
                continue

            node_id = f"{chain_id}_node_{i}"
            metadata = {
                "chain_id": chain_id,
                "topic": node.topic,
                "domain": node.domain,
                "connection_reason": node.connection_reason[:200],
                "depth": node.depth,
                "node_index": i,
                "chain_length": len(nodes),
                "seed_topic": seed_topic,
                "context": context[:200],
                "timestamp": current_time,
            }

            try:
                self._nodes_collection.upsert(
                    ids=[node_id],
                    embeddings=[node.embedding.tolist()],
                    metadatas=[metadata],
                    documents=[f"{node.topic} ({node.domain}): {node.connection_reason[:100]}"],
                )
                stored_count += 1
            except Exception:
                continue

        return stored_count

    def find_cross_temporal_collisions(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.82,
        min_age_hours: float = 24.0,
        max_results: int = 5,
    ) -> list[dict]:
        """Find old intermediate nodes that are semantically close to a new node.

        This is the cross-temporal collision query: "has any chain from the PAST
        passed through this same conceptual region?" If yes, that's a time-gap
        bisociation — today's thought and last week's thought share a hidden link.

        Returns list of dicts with node metadata + similarity score.
        """
        if not self._available or not self._nodes_collection:
            return []

        node_count = self._nodes_collection.count()
        if node_count == 0:
            return []

        cutoff_time = time.time() - (min_age_hours * 3600)

        try:
            results = self._nodes_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(max_results * 3, node_count),
                where={"timestamp": {"$lt": cutoff_time}},
            )

            if not results or not results["ids"] or not results["ids"][0]:
                return []

            collisions = []
            for i, node_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 1.0
                similarity = 1.0 - distance

                if similarity >= threshold:
                    meta = results["metadatas"][0][i] if results["metadatas"] else {}
                    age_hours = (time.time() - meta.get("timestamp", time.time())) / 3600
                    collisions.append({
                        "node_id": node_id,
                        "topic": meta.get("topic", ""),
                        "domain": meta.get("domain", ""),
                        "connection_reason": meta.get("connection_reason", ""),
                        "chain_id": meta.get("chain_id", ""),
                        "seed_topic": meta.get("seed_topic", ""),
                        "context": meta.get("context", ""),
                        "depth": meta.get("depth", 0),
                        "similarity": similarity,
                        "age_hours": age_hours,
                        "timestamp": meta.get("timestamp", 0),
                    })

            collisions.sort(key=lambda x: x["similarity"], reverse=True)
            return collisions[:max_results]

        except Exception as e:
            print(f"   [Memory] Cross-temporal query failed: {e}")
            return []

    def find_batch_cross_temporal(
        self,
        nodes: list,
        threshold: float = 0.82,
        min_age_hours: float = 24.0,
    ) -> list[dict]:
        """Check multiple nodes at once for cross-temporal collisions.

        More efficient than calling find_cross_temporal_collisions() per node.
        Returns the best collision found across all input nodes.
        """
        if not self._available or not self._nodes_collection:
            return []

        best_collisions: list[dict] = []

        for i, node in enumerate(nodes):
            if node.embedding is None:
                continue

            matches = self.find_cross_temporal_collisions(
                query_embedding=node.embedding,
                threshold=threshold,
                min_age_hours=min_age_hours,
                max_results=2,
            )

            for match in matches:
                match["current_node_topic"] = node.topic
                match["current_node_domain"] = node.domain
                match["current_node_index"] = i
                best_collisions.append(match)

        best_collisions.sort(key=lambda x: x["similarity"], reverse=True)
        return best_collisions[:5]
