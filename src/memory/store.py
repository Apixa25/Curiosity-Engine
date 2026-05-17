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

            count = self._collection.count()
            self._available = True
            print(f"   [Memory] ChromaDB ready — {count} chains in memory")
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
