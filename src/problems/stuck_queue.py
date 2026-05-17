"""
Stuck Queue — Background Problem Solver for Deep Thought Mode.

This is the Harman/Fadiman 1966 psychedelic problem-solving experiment,
made deterministic. The user declares a "stuck problem" — something they
can't solve by thinking directly. The engine then:

1. Generates BACKWARD chains from the problem target (working backward from
   the desired outcome toward increasingly fundamental mechanisms)
2. Stores those backward chains persistently
3. On every heartbeat, compares FORWARD chains (generated from ambient context)
   against stored BACKWARD chains (from the problem)
4. When a forward chain collides with a backward chain, the engine has
   found a hidden path FROM the user's daily life TO a solution component

This is the "walks in the park solve hard problems" phenomenon, mechanized.
The collision between "what you're doing right now" and "what you've been
stuck on for weeks" is often where breakthroughs hide.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from src.config.llm_adapter import LLMAdapter
from src.association_engine.tree_generator import AssociationTreeGenerator
from src.config.settings import AssociationTreeConfig, DeepThoughtConfig
from src.embeddings.provider import EmbeddingProvider
from src.models import AssociationChain, AssociationNode


@dataclass
class StuckProblem:
    """A problem the user is stuck on — the engine works on it in the background."""
    id: str = field(default_factory=lambda: f"prob-{uuid.uuid4().hex[:8]}")
    description: str = ""
    created_at: float = field(default_factory=time.time)
    backward_chains: list[AssociationChain] = field(default_factory=list)
    collision_count: int = 0
    last_chain_generation: float = 0.0
    active: bool = True

    @property
    def age_hours(self) -> float:
        return (time.time() - self.created_at) / 3600

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "created_at": self.created_at,
            "collision_count": self.collision_count,
            "last_chain_generation": self.last_chain_generation,
            "active": self.active,
            "backward_chain_count": len(self.backward_chains),
        }

    @classmethod
    def from_dict(cls, data: dict) -> StuckProblem:
        return cls(
            id=data["id"],
            description=data["description"],
            created_at=data["created_at"],
            collision_count=data.get("collision_count", 0),
            last_chain_generation=data.get("last_chain_generation", 0.0),
            active=data.get("active", True),
        )


BACKWARD_SYSTEM = (
    "You are Deep Thought working BACKWARD from a target problem. Instead of "
    "starting from a topic and exploring outward, you start from a DESIRED OUTCOME "
    "and work backward toward increasingly fundamental mechanisms, causes, and "
    "prerequisites. Each hop goes DEEPER into 'what would need to be true for this "
    "to work?' — moving toward root causes, hidden prerequisites, and foundational "
    "mechanisms that might connect to the user's daily experience. "
    "Be CONCRETE and SPECIFIC — name real phenomena, real mechanisms, real research."
)

BACKWARD_HOP1_TEMPLATE = """You are working BACKWARD from a problem target.

The user is stuck on: "{problem}"

Generate {n} associations that are one causal step BACKWARD from this problem:
- What are the hidden prerequisites for solving this?
- What fundamental mechanisms underlie this problem?
- What adjacent domains contain solutions people haven't tried?
- What would need to be true for a breakthrough to happen?

Work BACKWARD — from the desired outcome toward root causes and mechanisms.
Be SPECIFIC. Name real research, phenomena, biological pathways, physics principles.

Return ONLY a JSON array:
[{{"topic": "...", "domain": "...", "connection_reason": "..."}}]

Use one of these domains: {domains}"""

BACKWARD_HOP_TEMPLATE = """You are Deep Thought working BACKWARD through causal prerequisites.

The problem: "{problem}"
Backward chain so far: {chain_so_far}
Current node: "{current_topic}" (domain: {current_domain})

From "{current_topic}", generate {n} associations going FURTHER BACKWARD:
- What are the deeper prerequisites or root causes behind THIS mechanism?
- What everyday phenomena connect to this foundational layer?
- Cross into domains that seem far from the original problem
- The goal: reach mechanisms so fundamental they could connect to ANYTHING

Think: if someone walked past a {current_topic} on the street, what could that trigger
in their mind that would lead them back to solving "{problem}"?

Return ONLY a JSON array:
[{{"topic": "...", "domain": "...", "connection_reason": "..."}}]

Use one of these domains: {domains}"""


class StuckQueue:
    """Manages stuck problems and generates backward chains for collision detection.

    The queue persists problems to disk as JSON so they survive restarts.
    Backward chains are regenerated periodically (they're cheap to store in memory
    since we only need the intermediate node embeddings for collision detection).
    """

    def __init__(
        self,
        llm: LLMAdapter,
        tree_config: AssociationTreeConfig,
        dt_config: DeepThoughtConfig,
        embedder: EmbeddingProvider | None = None,
        persist_path: str = "data/problems.json",
    ):
        self.llm = llm
        self.tree_config = tree_config
        self.dt_config = dt_config
        self.embedder = embedder
        self.persist_path = Path(persist_path)
        self.problems: list[StuckProblem] = []
        self._load()

    @property
    def active_problems(self) -> list[StuckProblem]:
        return [p for p in self.problems if p.active]

    @property
    def has_problems(self) -> bool:
        return len(self.active_problems) > 0

    def add_problem(self, description: str) -> StuckProblem:
        """Add a new stuck problem to the queue."""
        problem = StuckProblem(description=description.strip())
        self.problems.append(problem)
        self._save()
        return problem

    def remove_problem(self, problem_id: str) -> bool:
        """Deactivate a problem (soft delete)."""
        for prob in self.problems:
            if prob.id == problem_id or prob.description.lower().startswith(problem_id.lower()):
                prob.active = False
                self._save()
                return True
        return False

    def get_all_backward_chains(self) -> list[AssociationChain]:
        """Get all backward chains across all active problems."""
        chains = []
        for prob in self.active_problems:
            chains.extend(prob.backward_chains)
        return chains

    def get_backward_chains_by_problem(self) -> dict[str, list[AssociationChain]]:
        """Get backward chains grouped by problem description (for collision labeling)."""
        result = {}
        for prob in self.active_problems:
            if prob.backward_chains:
                label = f"backward:{prob.description[:40]}"
                result[label] = prob.backward_chains
        return result

    async def generate_backward_chains(
        self,
        problem: StuckProblem,
        num_chains: int = 3,
        max_depth: int = 10,
    ) -> list[AssociationChain]:
        """Generate backward chains from a stuck problem.

        Uses the same tree generator infrastructure but with BACKWARD prompts
        that work from the desired outcome toward fundamental mechanisms.
        """
        from src.models import DOMAINS

        tree_gen = AssociationTreeGenerator(
            llm=self.llm,
            config=self.tree_config,
            embedder=self.embedder,
            deep_thought_mode=True,
            deep_thought_max_depth=max_depth,
            deep_thought_keep_per_level=self.dt_config.keep_per_level,
            deep_thought_min_domain_crossings=max(2, self.dt_config.min_domain_crossings - 1),
            deep_thought_invert_efficiency=self.dt_config.invert_efficiency,
            deep_thought_temperature_boost=self.dt_config.llm_temperature_boost,
        )

        tree_gen._backward_mode = True
        tree_gen._backward_problem = problem.description

        chains = await tree_gen.generate_tree(
            seed_topic=problem.description,
            max_depth=max_depth,
            keep_per_level=self.dt_config.keep_per_level,
            min_domain_crossings=2,
            mode_label="backward",
        )

        for chain in chains:
            chain.metadata = chain.metadata or {}
            chain.metadata["direction"] = "backward"
            chain.metadata["problem_id"] = problem.id
            chain.metadata["problem"] = problem.description

        problem.backward_chains = chains
        problem.last_chain_generation = time.time()
        self._save()

        return chains

    async def refresh_all_backward_chains(self) -> int:
        """Regenerate backward chains for all active problems.
        Returns total number of chains generated."""
        total = 0
        for prob in self.active_problems:
            regen_interval = 3600 * 4  # every 4 hours
            if time.time() - prob.last_chain_generation > regen_interval:
                chains = await self.generate_backward_chains(prob)
                total += len(chains)
        return total

    def record_collision(self, problem_id: str) -> None:
        """Record that a collision was found for this problem."""
        for prob in self.problems:
            if prob.id == problem_id:
                prob.collision_count += 1
                self._save()
                break

    def _save(self) -> None:
        """Persist problems to disk (chains are regenerated, not saved)."""
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = [p.to_dict() for p in self.problems]
        self.persist_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load(self) -> None:
        """Load problems from disk."""
        if self.persist_path.exists():
            try:
                data = json.loads(self.persist_path.read_text(encoding="utf-8"))
                self.problems = [StuckProblem.from_dict(d) for d in data]
            except (json.JSONDecodeError, KeyError):
                self.problems = []
