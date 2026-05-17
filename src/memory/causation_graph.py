"""
Personal Causation Graph — A Persistent Map of Your Entire Conceptual Universe.

Every intermediate node from every chain becomes a vertex in your personal graph.
Adjacent chain nodes become directed edges. Over time, this builds a complete map
of every concept your engine has ever traversed and every causal connection it has
found between them.

What makes this powerful:
1. SHORTEST PATH — Find unexpected connections between any two concepts through
   paths the engine discovered over weeks/months of operation
2. HUB NODES — Discover your "conceptual hubs" — nodes that connect the most
   disparate domains (these are your personal creative leverage points)
3. PERIPHERY SEEDS — Identify nodes at the far edges of your graph with few
   connections. These make excellent seeds for maximum collision potential
   (they represent conceptual frontier territory)
4. DOMAIN MAP — See which domains dominate your graph and where the bridges are
5. AFFINITY PROPAGATION — When you rate an interjection, that affinity flows
   through the graph edges, teaching the engine which conceptual PATHS you value

The graph persists to disk as JSON so it survives restarts and grows indefinitely.
"""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GraphNode:
    """A concept in the personal causation graph."""
    topic: str
    domain: str
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    visit_count: int = 1
    affinity: float = 0.0
    chain_ids: list[str] = field(default_factory=list)

    def touch(self, chain_id: str = "") -> None:
        """Record another visit to this node."""
        self.last_seen = time.time()
        self.visit_count += 1
        if chain_id and chain_id not in self.chain_ids:
            self.chain_ids.append(chain_id)

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "domain": self.domain,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "visit_count": self.visit_count,
            "affinity": self.affinity,
            "chain_ids": self.chain_ids[-20:],
        }

    @classmethod
    def from_dict(cls, data: dict) -> GraphNode:
        return cls(
            topic=data["topic"],
            domain=data["domain"],
            first_seen=data.get("first_seen", 0),
            last_seen=data.get("last_seen", 0),
            visit_count=data.get("visit_count", 1),
            affinity=data.get("affinity", 0.0),
            chain_ids=data.get("chain_ids", []),
        )


@dataclass
class GraphEdge:
    """A causal connection between two concepts."""
    source: str
    target: str
    weight: float = 1.0
    connection_reasons: list[str] = field(default_factory=list)
    chain_ids: list[str] = field(default_factory=list)
    traversal_count: int = 1

    def strengthen(self, reason: str = "", chain_id: str = "") -> None:
        """Strengthen this edge with another traversal."""
        self.traversal_count += 1
        self.weight = min(10.0, 1.0 + (self.traversal_count - 1) * 0.3)
        if reason and reason not in self.connection_reasons:
            self.connection_reasons.append(reason)
            if len(self.connection_reasons) > 5:
                self.connection_reasons = self.connection_reasons[-5:]
        if chain_id and chain_id not in self.chain_ids:
            self.chain_ids.append(chain_id)
            if len(self.chain_ids) > 10:
                self.chain_ids = self.chain_ids[-10:]

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "connection_reasons": self.connection_reasons,
            "chain_ids": self.chain_ids,
            "traversal_count": self.traversal_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> GraphEdge:
        return cls(
            source=data["source"],
            target=data["target"],
            weight=data.get("weight", 1.0),
            connection_reasons=data.get("connection_reasons", []),
            chain_ids=data.get("chain_ids", []),
            traversal_count=data.get("traversal_count", 1),
        )


@dataclass
class PathResult:
    """A shortest-path query result."""
    source: str
    target: str
    path: list[str]
    total_weight: float
    hop_count: int
    domains_crossed: list[str]

    def __str__(self) -> str:
        path_str = " → ".join(self.path)
        return f"{path_str} ({self.hop_count} hops, weight: {self.total_weight:.2f})"


class CausationGraph:
    """Persistent personal concept graph.

    Built incrementally from every chain the engine generates. Nodes are
    normalized by lowercased topic so "Blue Light" and "blue light" merge.
    Edges are directed (A→B means the engine once went from concept A to B).
    """

    def __init__(self, persist_path: str = "data/memory/causation_graph.json"):
        self.persist_path = Path(persist_path)
        self.nodes: dict[str, GraphNode] = {}
        self.edges: dict[str, GraphEdge] = {}
        self._adjacency: dict[str, set[str]] = defaultdict(set)
        self._reverse_adjacency: dict[str, set[str]] = defaultdict(set)
        self._load()

    def _node_key(self, topic: str) -> str:
        """Normalize topic to a graph key."""
        return topic.lower().strip()

    def _edge_key(self, source: str, target: str) -> str:
        return f"{self._node_key(source)}||{self._node_key(target)}"

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def add_chain(self, nodes: list, chain_id: str = "") -> int:
        """Add all nodes and edges from a chain to the graph.

        Each node becomes a graph vertex, each pair of adjacent nodes
        becomes a directed edge. Returns number of new nodes added.
        """
        new_nodes = 0

        for i, node in enumerate(nodes):
            key = self._node_key(node.topic)

            if key in self.nodes:
                self.nodes[key].touch(chain_id)
            else:
                self.nodes[key] = GraphNode(
                    topic=node.topic,
                    domain=node.domain,
                    chain_ids=[chain_id] if chain_id else [],
                )
                new_nodes += 1

            if i > 0:
                prev_node = nodes[i - 1]
                source_key = self._node_key(prev_node.topic)
                target_key = key
                edge_key = self._edge_key(prev_node.topic, node.topic)

                if edge_key in self.edges:
                    self.edges[edge_key].strengthen(
                        reason=node.connection_reason,
                        chain_id=chain_id,
                    )
                else:
                    self.edges[edge_key] = GraphEdge(
                        source=source_key,
                        target=target_key,
                        connection_reasons=[node.connection_reason] if node.connection_reason else [],
                        chain_ids=[chain_id] if chain_id else [],
                    )

                self._adjacency[source_key].add(target_key)
                self._reverse_adjacency[target_key].add(source_key)

        return new_nodes

    def propagate_rating(self, chain_nodes: list, rating: int) -> None:
        """Propagate a user rating through the graph edges of a chain.

        High ratings increase affinity on nodes along the path.
        Low ratings decrease it. The effect decays with distance from endpoints.
        """
        if not chain_nodes:
            return

        affinity_delta = (rating - 3) * 0.15
        n = len(chain_nodes)

        for i, node in enumerate(chain_nodes):
            key = self._node_key(node.topic)
            if key in self.nodes:
                decay = 1.0 - (min(i, n - 1 - i) / max(n, 1)) * 0.5
                self.nodes[key].affinity += affinity_delta * decay
                self.nodes[key].affinity = max(-2.0, min(2.0, self.nodes[key].affinity))

    def shortest_path(self, source_topic: str, target_topic: str) -> PathResult | None:
        """Find shortest path between two concepts using BFS.

        Uses unweighted BFS for simplicity (all edges = 1 hop).
        Returns None if no path exists.
        """
        source_key = self._node_key(source_topic)
        target_key = self._node_key(target_topic)

        if source_key not in self.nodes or target_key not in self.nodes:
            return None

        if source_key == target_key:
            return PathResult(
                source=source_topic, target=target_topic,
                path=[source_topic], total_weight=0, hop_count=0, domains_crossed=[],
            )

        visited = {source_key}
        queue = [(source_key, [source_key])]

        while queue:
            current, path = queue.pop(0)
            for neighbor in self._adjacency.get(current, set()):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                new_path = path + [neighbor]

                if neighbor == target_key:
                    topics = [self.nodes[k].topic for k in new_path if k in self.nodes]
                    domains = []
                    for k in new_path:
                        if k in self.nodes and self.nodes[k].domain not in domains:
                            domains.append(self.nodes[k].domain)
                    return PathResult(
                        source=source_topic, target=target_topic,
                        path=topics, total_weight=len(new_path) - 1,
                        hop_count=len(new_path) - 1, domains_crossed=domains,
                    )

                queue.append((neighbor, new_path))

        return None

    def hub_nodes(self, top_n: int = 10) -> list[tuple[str, int, str]]:
        """Find the most connected nodes (highest degree centrality).

        Hub nodes connect the most disparate concepts — they're your
        personal creative leverage points. Returns (topic, degree, domain).
        """
        degree: dict[str, int] = {}
        for key in self.nodes:
            out_degree = len(self._adjacency.get(key, set()))
            in_degree = len(self._reverse_adjacency.get(key, set()))
            degree[key] = out_degree + in_degree

        sorted_nodes = sorted(degree.items(), key=lambda x: x[1], reverse=True)

        results = []
        for key, deg in sorted_nodes[:top_n]:
            if key in self.nodes:
                results.append((self.nodes[key].topic, deg, self.nodes[key].domain))
        return results

    def periphery_nodes(self, top_n: int = 10) -> list[tuple[str, str]]:
        """Find nodes at the far edges of the graph with fewest connections.

        These make excellent seeds for maximum collision potential — they're
        frontier territory that hasn't been well-explored yet. Only returns
        nodes with at least one connection (isolated nodes are noise).

        Returns (topic, domain) tuples.
        """
        degree: dict[str, int] = {}
        for key in self.nodes:
            out_degree = len(self._adjacency.get(key, set()))
            in_degree = len(self._reverse_adjacency.get(key, set()))
            total = out_degree + in_degree
            if total > 0:
                degree[key] = total

        sorted_nodes = sorted(degree.items(), key=lambda x: x[1])

        results = []
        for key, _ in sorted_nodes[:top_n * 2]:
            if key in self.nodes:
                results.append((self.nodes[key].topic, self.nodes[key].domain))

        random.shuffle(results)
        return results[:top_n]

    def high_affinity_nodes(self, top_n: int = 10) -> list[tuple[str, float, str]]:
        """Find nodes the user has implicitly shown preference for.

        Returns (topic, affinity, domain) sorted by affinity descending.
        """
        sorted_nodes = sorted(
            self.nodes.items(),
            key=lambda x: x[1].affinity,
            reverse=True,
        )
        results = []
        for key, node in sorted_nodes[:top_n]:
            if node.affinity > 0:
                results.append((node.topic, node.affinity, node.domain))
        return results

    def domain_map(self) -> dict[str, int]:
        """Count nodes per domain — your personal concept universe breakdown."""
        counts: dict[str, int] = defaultdict(int)
        for node in self.nodes.values():
            counts[node.domain] += 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    def bridge_edges(self, top_n: int = 10) -> list[tuple[str, str, str, str]]:
        """Find edges that bridge between different domains.

        These are the creative cross-domain connections — the most interesting
        edges in your graph. Returns (source_topic, target_topic, source_domain, target_domain).
        """
        bridges = []
        for edge in self.edges.values():
            source_node = self.nodes.get(edge.source)
            target_node = self.nodes.get(edge.target)
            if source_node and target_node and source_node.domain != target_node.domain:
                bridges.append((
                    source_node.topic,
                    target_node.topic,
                    source_node.domain,
                    target_node.domain,
                    edge.traversal_count,
                ))

        bridges.sort(key=lambda x: x[4], reverse=True)
        return [(s, t, sd, td) for s, t, sd, td, _ in bridges[:top_n]]

    def get_periphery_seed(self) -> str | None:
        """Get a random periphery node topic for use as a multi-seed source.

        Periphery nodes have the most collision potential because they sit
        at the edges of the user's known concept universe — chains from here
        are most likely to explore genuinely new territory.
        """
        candidates = self.periphery_nodes(top_n=20)
        if not candidates:
            return None
        topic, _ = random.choice(candidates)
        return topic

    def get_high_affinity_seed(self) -> str | None:
        """Get a random high-affinity node for seeds the user is likely to enjoy."""
        candidates = self.high_affinity_nodes(top_n=15)
        if not candidates:
            return None
        topic, _, _ = random.choice(candidates)
        return topic

    def stats(self) -> dict:
        """Summary statistics for the graph."""
        domains = self.domain_map()
        hubs = self.hub_nodes(top_n=5)
        avg_degree = 0.0
        if self.nodes:
            total_edges = sum(
                len(self._adjacency.get(k, set())) for k in self.nodes
            )
            avg_degree = total_edges / len(self.nodes)

        return {
            "nodes": self.node_count,
            "edges": self.edge_count,
            "domains": len(domains),
            "top_domains": list(domains.items())[:5],
            "avg_degree": avg_degree,
            "top_hubs": hubs,
        }

    def _save(self) -> None:
        """Persist the graph to disk as JSON."""
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": {k: v.to_dict() for k, v in self.edges.items()},
        }
        self.persist_path.write_text(json.dumps(data), encoding="utf-8")

    def _load(self) -> None:
        """Load the graph from disk."""
        if not self.persist_path.exists():
            return

        try:
            data = json.loads(self.persist_path.read_text(encoding="utf-8"))
            for key, node_data in data.get("nodes", {}).items():
                self.nodes[key] = GraphNode.from_dict(node_data)
            for key, edge_data in data.get("edges", {}).items():
                edge = GraphEdge.from_dict(edge_data)
                self.edges[key] = edge
                self._adjacency[edge.source].add(edge.target)
                self._reverse_adjacency[edge.target].add(edge.source)
        except (json.JSONDecodeError, KeyError, TypeError):
            self.nodes = {}
            self.edges = {}

    def save(self) -> None:
        """Public save method — call after batch operations."""
        self._save()
