# 🧠🔮 Deep Thought Build Guide — From Logical Chains to Butterfly-Effect Discoveries

**Author:** Steven Sills II  
**Date:** May 2026  
**Status:** Active development roadmap  

---

## The North Star

> "Would I have invented PCR if I hadn't taken LSD? I seriously doubt it."  
> — Kary Mullis, 1993 Nobel Prize in Chemistry

The Creativity Engine currently produces **interesting, defensible, surprising** causal chains — a friend who notices cool connections. The goal of this build guide is to evolve it into **Deep Thought** — a personal oracle that surfaces hidden causal linkages so distant, so complex, that they feel **magical and earth-shattering** to a human observer.

We are building deterministic LSD: hyper-connectivity between normally separate concept networks, producing hypotheses no human mind could trace alone.

**Key philosophical constraint:** We are NOT adding randomness or hallucination. We are making the **real causal chains longer and more structurally complex** — so complex that the same hidden-causation principle ("free will = causation complex enough to be invisible") produces outputs indistinguishable from superintelligent insight rather than mere creative friendship.

---

## Table of Contents

1. [Current State Assessment](#1-current-state-assessment)
2. [The Fundamental Shift: Chain Topology](#2-the-fundamental-shift-chain-topology)
3. [Phase 1: Multi-Seed Parallel Chains](#3-phase-1-multi-seed-parallel-chains-the-lsd-entropy)
4. [Phase 2: The Collision Engine](#4-phase-2-the-collision-engine-bisociation-detection)
5. [Phase 3: Bidirectional Chaining & Stuck Problems](#5-phase-3-bidirectional-chaining--stuck-problems)
6. [Phase 4: Deep Thought Scoring](#6-phase-4-deep-thought-scoring)
7. [Phase 5: Hypothesis-Style Output](#7-phase-5-hypothesis-style-output)
8. [Phase 6: Cross-Temporal Memory Collisions](#8-phase-6-cross-temporal-memory-collisions)
9. [Phase 7: The Personal Causation Graph](#9-phase-7-the-personal-causation-graph)
10. [Research Foundation](#10-research-foundation)
11. [Configuration & Modes](#11-configuration--modes)
12. [File-by-File Change Map](#12-file-by-file-change-map)
13. [Prioritized Build Order](#13-prioritized-build-order)
14. [Epistemological Safeguards](#14-epistemological-safeguards)

---

## 1. Current State Assessment

### What We Have (and why it's solid)

| Component | File | What It Does Now | Rating |
|---|---|---|---|
| Association Tree | `src/association_engine/tree_generator.py` | Ternary branching, 4-7 hops, depth-first pruning, embedding-aware distance | Great foundation |
| Interest Scorer | `src/scoring/interest_scorer.py` | 5-metric formula with real cosine distance + Kenett efficiency bonus | Solid |
| Bridge Builder | `src/bridge_builder/builder.py` | Natural interjections, excitement tiers, hides machinery | Perfect |
| Personalities | `src/bridge_builder/personalities.py` | 5 rotating archetypes, domain-weighted selection | Good |
| Memory Store | `src/memory/store.py` | ChromaDB persistent storage, ratings, full scoring breakdown | Good |
| Incubation | `src/memory/incubation.py` | Re-score shelved chains hourly, daily reflections | Good |
| Profile | `src/memory/profile.py` | Auto-generated user persona from rating history | Good |
| Analytics | `src/memory/analytics.py` | AHA! rate, Mednick/Kenett metrics, trend detection | Good |
| Transparency | `src/main.py` | Full causal chain reveal on demand | Great |
| Multimodal Input | `src/input_pipeline/` | Vision, audio, screen, prosody, novelty weighting | Excellent |
| Embeddings | `src/embeddings/provider.py` | OpenAI + local, caching, cosine math | Good |

### Why It Produces "Interesting Friend" But Not "Deep Thought"

1. **Single-chain topology** — One tree from one seed per heartbeat. LSD-mode creativity requires many parallel chains from different seeds finding collision points.

2. **Forward-only chaining** — The tree always starts from your context and goes outward. Never starts from a *target* (like "cure cancer") and chains backward to collide with your life.

3. **Too few hops** — Max depth 7 produces chains a clever human could trace. Depth 15-30 produces chains NO human can trace — the causation becomes genuinely invisible.

4. **Scorer rewards "tellable stories"** — Bridgeability at 0.15 weight means we only surface chains where the LLM can construct a narrative. Deep Thought outputs hypotheses that may resist simple narration — the chain is the proof, not the bridge.

5. **No collision detection** — We never compare intermediate nodes across different chains. This is where bisociation (Koestler) lives — the collision point between two independent causal threads.

6. **No "stuck problem" mode** — The 1966 Harman/Fadiman study worked because experts brought *specific unsolved problems*. Our engine only associates from ambient context.

7. **Incubation is temporal, not structural** — Re-scoring one chain against new context is good. But detecting when *two stored chains from different sessions share a hidden intermediate concept* — that's the earth-shattering unlock.

---

## 2. The Fundamental Shift: Chain Topology

The key insight: **LSD doesn't make individual thoughts wilder. It makes SEPARATE neural networks fire simultaneously and find collision points.**

### Current Architecture (Single Linear Chain)

```
Seed → Hop 1 → Hop 2 → Hop 3 → Hop 4 → Endpoint
(one seed, one path, each hop defensible, total chain traceable by humans)
```

### Deep Thought Architecture (Multi-Seed Collision Network)

```
Seed A (current context) ──→──→──→──→──→──→──→──→───┐
                                                      │
Seed B (random memory)   ──→──→──→──→──→──→──→──→───┤── COLLISION ──→ Hypothesis
                                                      │
Seed C (stuck problem)   ←──←──←──←──←──←──←──←───┘
           (backward chain from target)
```

The collision point is where two (or more) independent causal threads share a **hidden intermediate concept** in embedding space. This is computationally equivalent to LSD's hyper-connectivity: normally anti-correlated brain networks communicating through a shared node.

---

## 3. Phase 1: Multi-Seed Parallel Chains (The LSD Entropy)

**Goal:** Instead of one tree per heartbeat, generate 5-10 trees from different seed perspectives simultaneously.

### New File: `src/association_engine/multi_seed.py`

This module orchestrates parallel chain generation from multiple seeds:

**Seed sources (5-10 per cycle):**

| Seed Source | Where It Comes From | Why |
|---|---|---|
| Current context | `ContextAssembler` output | Your immediate world |
| Random memory node | Random `StoredChain` from ChromaDB | Something from days/weeks ago — temporal collision potential |
| Stuck problem | New "problems queue" (user-defined) | The 1966 study: experts with specific unsolved problems |
| Inverse context | LLM-generated "opposite" of current context | Force anti-correlated associations (DMN disruption) |
| Web signal | Trending topic from Tavily | External world collision with your internal world |
| Personality-driven | Each personality archetype suggests a seed from their domain | Ensure domain diversity |

**Key design decisions:**

- Each seed gets its OWN independent tree (not shared branches)
- Trees run in parallel via `asyncio.gather`
- ALL intermediate nodes get embedded (not just endpoints)
- Chains are stored with a `batch_id` linking them to the same heartbeat cycle
- Pass all generated chains into the Collision Engine (Phase 2) before scoring

### Changes to `src/association_engine/tree_generator.py`

**New prompts for Deep Thought mode:**

```python
DEEP_THOUGHT_HOP_TEMPLATE = """You are computing hidden causal linkages in the universe.
All things are causally connected — the question is how DISTANT the linkage is.

From "{current_topic}" (chain so far: {chain_so_far}), generate {n} associations via:
- Butterfly-effect reasoning: how could a tiny change HERE cascade to a massive outcome THERE?
- Cross-sensory/cross-scale jumps: micro to macro, physical to social, chemical to cultural
- Hidden mechanism chains: A causes B through a mechanism most humans don't know exists
- Koestler bisociation: find where two completely unrelated systems share a structural pattern

Be CONCRETE and SPECIFIC. Name real phenomena, real mechanisms, real research.
Do NOT be vague or abstract. Each association should name a specific thing that exists.

Return ONLY a JSON array:
[{{"topic": "...", "domain": "...", "connection_reason": "..."}}]

Use one of these domains: {domains}"""
```

**Config changes (`AssociationTreeConfig`):**

```python
# Normal mode (current defaults)
max_depth: 7
keep_per_level: 3
branching_factor: 3
min_domain_crossings: 1

# Deep Thought mode
max_depth: 15          # up to 30 in "ultimate question" mode
keep_per_level: 5      # wider exploration
branching_factor: 3    # same branching, but deeper
min_domain_crossings: 4 # force more boundary-crossing
```

**Pruning changes for Deep Thought mode:**

In `_rank_for_pruning`, the current logic rewards chains that reach far in few hops (Kenett efficiency). For Deep Thought mode, **invert this**: reward chains that take MANY hops to cover distance — because each hop is another causal link in the invisible chain. More hops = more hidden causation = more "magical" output.

```python
# Normal mode (current): efficiency_bonus rewards fewer hops for same distance
# Deep Thought mode: inefficiency_bonus rewards MORE hops
if deep_thought_mode:
    inefficiency = hops / max(raw_distance, 0.01)  # more hops per unit distance = better
    score = (raw_distance * 0.5) + (min(1.0, inefficiency * 0.1) * 0.5)
```

---

## 4. Phase 2: The Collision Engine (Bisociation Detection)

**Goal:** After parallel chains are generated, detect when independent chains share hidden intermediate concepts in embedding space.

### New File: `src/association_engine/collision_engine.py`

This is the core Deep Thought innovation. It takes all chains from a cycle (or from memory) and finds **bisociation points** — places where two independent causal threads pass through the same conceptual region.

**Algorithm:**

```
1. Collect all intermediate nodes from all parallel chains
2. Build a node_embeddings matrix (N nodes × embedding_dim)
3. Compute pairwise cosine similarity across nodes from DIFFERENT chains
4. For each pair with similarity > COLLISION_THRESHOLD (e.g., 0.82):
   a. Verify they come from different seed sources
   b. Extract the two parent chains
   c. Build a CollisionResult containing:
      - Chain A (seed → ... → collision_node_A)
      - Chain B (seed → ... → collision_node_B)
      - The collision concept (averaged/summarized)
      - The combined causal distance (hops_A + hops_B + domain_crossings_combined)
      - Collision confidence (based on embedding similarity at the collision point)
```

### New Data Model: `CollisionResult`

```python
@dataclass
class CollisionResult:
    """Two independent causal threads that share a hidden intermediate concept."""
    chain_a: AssociationChain         # Forward chain from seed A
    chain_b: AssociationChain         # Forward chain from seed B (or backward from target)
    collision_node_a: AssociationNode # The node in chain A near the collision
    collision_node_b: AssociationNode # The node in chain B near the collision
    collision_concept: str            # The shared concept (LLM-synthesized from both nodes)
    collision_similarity: float       # Cosine similarity at the collision point
    total_causal_distance: float      # Combined semantic distance across both chains
    total_hops: int                   # Combined hop count
    total_domain_crossings: int       # Combined domain crossings
    hypothesis: str | None = None     # Generated hypothesis (filled by bridge builder)
```

**Why this is the LSD moment:**

In the brain under LSD, the default mode network (DMN) loses its gatekeeping role. Brain regions that normally suppress each other fire simultaneously. When two separate activation patterns overlap at a shared neural ensemble, that's bisociation — the creative "aha!".

The Collision Engine does exactly this computationally: two independent chains (separate "brain networks") are checked for overlap in embedding space (shared "neural ensemble"). The collision point is invisible to any single chain — it only appears when you compare across chains.

**Collision scoring (feeds into Phase 4):**

The most interesting collisions have:
- **Maximum total causal distance** (seed A and seed B were extremely far apart)
- **Maximum total domain crossings** (the chains crossed many fields)
- **High collision similarity** (the intermediate nodes are genuinely about the same thing)
- **Neither chain endpoint is the collision** (the shared concept is HIDDEN, not at the surface)

---

## 5. Phase 3: Bidirectional Chaining & Stuck Problems

**Goal:** Let the user define "stuck problems" — things they're trying to solve. The engine chains BACKWARD from the target and looks for collisions with FORWARD chains from daily life.

### New File: `src/problems/stuck_queue.py`

```python
@dataclass
class StuckProblem:
    """A problem the user wants the engine to work on in the background."""
    id: str
    problem_statement: str           # "How do I cure colon cancer?"
    target_concepts: list[str]       # ["colon cancer", "tumor suppression", "gut health"]
    backward_chains: list[str]       # IDs of stored backward chains
    created_at: float
    last_chain_generation: float     # when we last generated backward chains
    collision_count: int = 0         # how many collisions detected so far
    active: bool = True
```

**How bidirectional chaining works:**

```
User says: "solve colon cancer"

Engine generates BACKWARD chains from "colon cancer":
  colon cancer ← gut inflammation ← microbiome disruption ← circadian rhythm ← blue light exposure ← ...
  colon cancer ← tumor suppression ← p53 pathway ← DNA repair ← UV radiation ← ...
  colon cancer ← diet ← fiber fermentation ← short-chain fatty acids ← ...

These backward chains are stored permanently in ChromaDB.

Every heartbeat, FORWARD chains from current context are generated:
  kitchen remodel → paint colors → wavelength of blue → blue light spectrum → ...
  LED project → light frequencies → circadian research → ...

COLLISION DETECTED:
  Forward chain passes through "blue light / circadian"
  Backward chain passes through "circadian rhythm / gut inflammation"
  
  → Hypothesis: "The blue paint in your kitchen may affect circadian-regulated
    gut processes through nightly blue light exposure. Research links circadian
    disruption to inflammatory gut conditions that increase colon cancer risk."
```

**This is the "paint your cabinet blue → cure cancer" chain.** Each individual hop is defensible. But the total chain — connecting kitchen paint to cancer through 15+ intermediate hops across 8 domains — is invisible to human cognition. Only the engine can see it because it's tracking both directions simultaneously.

### New Commands in `src/main.py`

```
'solve <problem>'  → Add a stuck problem to the background queue
'problems'         → List active stuck problems and collision counts
'forget <problem>' → Remove a stuck problem from the queue
```

### Integration with Heartbeat Cycle

In `_on_heartbeat` in `main.py`:

```python
# After normal forward chains are generated:
if self.stuck_queue.has_active_problems():
    # Generate backward chains from each stuck problem (if not recently done)
    await self.stuck_queue.refresh_backward_chains(self.tree_gen)
    
    # Check for collisions between today's forward chains and stored backward chains
    collisions = await self.collision_engine.detect_cross_directional(
        forward_chains=today_forward_chains,
        backward_chains=self.stuck_queue.all_backward_chains(),
    )
    
    if collisions:
        best = max(collisions, key=lambda c: c.total_causal_distance)
        hypothesis = await self.bridge.build_hypothesis(best)
        # Deliver as a special "Deep Thought" interjection
```

---

## 6. Phase 4: Deep Thought Scoring

**Goal:** A modified scoring system that evaluates collision quality rather than single-chain interestingness.

### Changes to `src/scoring/interest_scorer.py`

**New method: `score_collision`**

```python
async def score_collision(
    self,
    collision: CollisionResult,
    context: ContextSnapshot,
) -> CollisionScore:
    """Score a collision between two independent chains."""
    
    # 1. Causal Depth — How many total hops across both chains?
    #    More hops = more hidden causation = more "magical"
    causal_depth = self._compute_causal_depth(collision)
    
    # 2. Seed Distance — How far apart were the original seeds?
    #    Farther = more surprising that they connect
    seed_distance = self._compute_seed_distance(collision)
    
    # 3. Collision Hiddenness — Is the collision point buried in the middle of chains?
    #    vs. at the surface endpoints? Hidden = more magical
    hiddenness = self._compute_hiddenness(collision)
    
    # 4. Cross-Domain Span — How many total domain boundaries crossed?
    domain_span = self._compute_domain_span(collision)
    
    # 5. Mechanism Specificity — Does the collision involve a specific mechanism?
    #    (LLM check: "Is there a known causal mechanism linking these?")
    mechanism = await self._compute_mechanism_specificity(collision)
    
    # 6. Testability — Can this collision be turned into a testable hypothesis?
    testability = await self._compute_testability(collision)
```

**New scoring weights for Deep Thought mode:**

```
collision_score = (causal_depth      × 0.20)  — reward invisible complexity
               + (seed_distance      × 0.20)  — reward extreme starting distance
               + (hiddenness         × 0.15)  — reward buried collision points
               + (domain_span        × 0.15)  — reward crossing many fields
               + (mechanism          × 0.15)  — reward real causal mechanisms
               + (testability        × 0.15)  — reward actionable hypotheses
```

Notice: **no bridgeability**. The hypothesis IS the output — we don't need to "bridge" back to conversation. We need to state the hidden link and how to test it.

### New Data Model: `CollisionScore`

Add to `src/models.py`:

```python
@dataclass
class CollisionScore:
    causal_depth: float
    seed_distance: float
    hiddenness: float
    domain_span: float
    mechanism_specificity: float
    testability: float
    total: float
```

---

## 7. Phase 5: Hypothesis-Style Output

**Goal:** Deep Thought mode doesn't output "fun facts" — it outputs **testable hypotheses** with confidence levels.

### Changes to `src/bridge_builder/builder.py`

**New method: `build_hypothesis`**

```python
HYPOTHESIS_SYSTEM = (
    "You are Deep Thought — a personal oracle that has just discovered a hidden "
    "causal linkage between things that no human could connect on their own. "
    "You speak with calm authority, like someone who just finished a 7.5-million-year "
    "computation and found the answer is '42.' You're not showing off. You're sharing "
    "something profound that you think the user needs to hear."
)

HYPOTHESIS_TEMPLATE = """You discovered a hidden causal connection:

Chain A started from: "{seed_a}" and reached: "{collision_a}"
Chain B started from: "{seed_b}" and reached: "{collision_b}"
The hidden link between them: "{collision_concept}"

Total causal hops: {total_hops} across {domain_crossings} domains
Web-grounded facts found: {facts}

Write a hypothesis in this format:
1. THE CLAIM: One bold sentence stating the hidden connection (like "42")
2. THE CHAIN: 2-3 sentences explaining the butterfly-effect cascade (just enough for the user to see it's not random, but not so much they can fully trace it)
3. THE TEST: One sentence saying how you could verify or explore this further
4. CONFIDENCE: Low / Medium / High (based on how many hops are web-grounded)

Sound like a friend who just had a profound realization — not a research paper.
Do NOT use markdown, asterisks, or formatting. Plain conversational text.
Do NOT start with "I" — start with the insight itself."""
```

**Output format example:**

```
══════════════════════════════════════════════════════════
🔮 DEEP THOUGHT SPEAKS  [hypothesis | confidence: medium]

   "The blue paint on your third kitchen cabinet might be nudging your
   body toward colon cancer risk every evening. Here's the chain I see:
   that specific blue reflects at 470nm — peak circadian disruption
   wavelength. Your kitchen light hits it while you're cooking dinner,
   right when your body is trying to wind down melatonin production.
   Melatonin doesn't just regulate sleep — it directly modulates gut
   motility and bacterial diversity. There's a 2019 paper showing
   circadian-disrupted mice have 3x the rate of colonic inflammation.
   
   Test it: Cover the cabinet for two weeks and track your sleep quality.
   If your sleep improves, the cascade is real."

══════════════════════════════════════════════════════════
```

### New Excitement Tier

Add a new tier above `mind_blown`:

```python
{
    "name": "deep_thought",
    "min_score": 0.0,   # triggered by collision, not single-chain score
    "tone": "calm, profound, slightly amused — like you just solved a cosmic puzzle",
    "length": "4-8 sentences — claim, chain, test, confidence",
    "energy": "Not excited. CERTAIN. The quiet confidence of an entity that computed for 7.5 million years.",
}
```

---

## 8. Phase 6: Cross-Temporal Memory Collisions

**Goal:** Detect when today's chain collides with something from weeks ago. This is the "7.5 million years" slow computation.

### Changes to `src/memory/store.py`

**New method: `find_cross_temporal_collisions`**

Every time a new chain is stored, check ALL its intermediate nodes against ALL intermediate nodes from past sessions:

```python
def find_cross_temporal_collisions(
    self,
    new_chain_nodes: list[tuple[str, np.ndarray]],  # (topic, embedding) pairs
    min_age_hours: float = 24.0,                     # only collide with chains > 24h old
    similarity_threshold: float = 0.82,
) -> list[dict]:
    """Find intermediate nodes from old chains that are similar to new chain nodes."""
```

**This is the earth-shattering moment:** You're working on your LED project today. The engine generates forward chains about LED light spectra. One intermediate node is "470nm blue light spectrum." 

Three weeks ago, you were researching gut health after a doctor's visit. The engine generated chains about colon health. One intermediate node was "circadian regulation of gut motility."

The embedding of "470nm blue light spectrum" and "circadian regulation" are similar in vector space (both relate to blue light's biological effects). **COLLISION DETECTED across a 3-week gap.**

The engine surfaces: "Remember when you were looking into gut health? I just realized your LED project connects to that through circadian blue-light pathways..."

### Changes to `src/memory/incubation.py`

Add cross-temporal collision checks to the existing hourly re-score cycle. After re-scoring individual chains, also run collision detection across the full memory store.

---

## 9. Phase 7: The Personal Causation Graph

**Goal:** Build a persistent graph of YOUR causal universe — every context, every chain, every collision, every rating — and use it to find connections that are personally meaningful.

### New File: `src/memory/causation_graph.py`

```python
@dataclass
class CausalNode:
    """A node in the user's personal causation graph."""
    concept: str
    embedding: np.ndarray
    domain: str
    first_seen: float          # when this concept first appeared in your life
    last_seen: float           # most recent appearance
    occurrence_count: int      # how many times it's appeared
    source_contexts: list[str] # what you were doing when this concept appeared
    connections: list[str]     # IDs of CausalNodes this connects to
    user_affinity: float       # how much you seem to care about this (from ratings)
```

**Graph construction:**
- Every intermediate node from every chain becomes a `CausalNode`
- Edges are created between adjacent nodes in chains
- User ratings propagate affinity through the graph
- The graph grows over weeks/months as you use the engine

**Graph queries for Deep Thought mode:**
- "Find the shortest path between `concept_A` and `concept_B` in my personal graph" (these become hypotheses)
- "Find all concepts within 3 hops of my stuck problem that also connect to my current context"
- "What concept in my graph has the most connections to different domains?" (personal hub nodes = where YOUR hidden causation lives)

---

## 10. Research Foundation

Everything in this guide is grounded in published research:

| Concept | Research | How We Use It |
|---|---|---|
| **Remote Associates** | Mednick (1962) | Creativity = connecting distant concepts. Our chains already do this; Deep Thought extends the distance. |
| **Flat Semantic Networks** | Kenett et al. (2014-2018) | Creative people have more interconnected concept networks. The Collision Engine builds this computationally. |
| **Bisociation** | Koestler (1964) | Creativity happens when two independent "matrices of thought" collide. Collision detection is literal bisociation. |
| **Entropic Brain** | Carhart-Harris (2014) | LSD increases brain entropy, allowing more possible states. Multi-seed parallel chaining increases computational entropy. |
| **Hyper-Connectivity** | Luppi et al. (2021) | LSD enables normally anti-correlated networks to communicate. Cross-chain collision detection enables normally separate concept chains to communicate. |
| **DMN Disruption** | Mason et al. (2021) | Psychedelics suppress the "default mode" (normal/expected filter). Deep Thought mode suppresses bridgeability (the "can I tell a story?" filter). |
| **Creative Problem Solving** | Harman/Fadiman (1966) | 27 professionals, 44/48 breakthroughs on LSD. Key: they brought SPECIFIC problems. Our "stuck problem" queue replicates this. |
| **Small World Networks** | Watts & Strogatz (1998) | ~6 degrees of separation in concept space. Our chains go 15-30 hops — far past the small-world threshold into "no human can trace this" territory. |
| **Butterfly Effect** | Lorenz (1963) | Tiny changes cascade to massive outcomes through deterministic but complex systems. Each hop is deterministic; the total chain is unpredictable to humans. |

---

## 11. Configuration & Modes

### New Config Section in `config.example.yaml`

```yaml
creativity_mode: "normal"   # "normal", "deep_thought", or "ultimate_question"

deep_thought:
  enabled: false
  parallel_seeds: 5              # how many independent chains to generate per heartbeat
  max_depth: 15                  # chain depth (up to 30 in ultimate_question mode)
  keep_per_level: 5              # wider exploration at each depth level
  min_domain_crossings: 4        # force more boundary-crossing
  collision_threshold: 0.82      # cosine similarity for collision detection
  cross_temporal_enabled: true   # check new chains against old memory
  cross_temporal_min_age_hours: 24
  invert_efficiency: true        # reward MORE hops (opposite of normal Kenett scoring)
  llm_temperature_boost: 0.2    # add to base temperature for Deep Thought hops
  
  scoring_weights:
    causal_depth: 0.20
    seed_distance: 0.20
    hiddenness: 0.15
    domain_span: 0.15
    mechanism_specificity: 0.15
    testability: 0.15

stuck_problems:
  enabled: false
  max_active_problems: 5
  backward_chain_refresh_hours: 4   # regenerate backward chains every N hours
  backward_chain_depth: 15
```

### CLI Flags

```bash
# Normal mode (current behavior — friendly creative companion)
python -m src.main --live

# Deep Thought mode (parallel chains, collision detection, hypothesis output)
python -m src.main --live --mode deep_thought

# Ultimate Question mode (dedicated to solving stuck problems, longer chains, persistent)
python -m src.main --live --mode ultimate_question
```

---

## 12. File-by-File Change Map

### New Files to Create

| File | Purpose |
|---|---|
| `src/association_engine/multi_seed.py` | Orchestrates parallel chain generation from multiple seed sources |
| `src/association_engine/collision_engine.py` | Detects bisociation points between independent chains |
| `src/problems/stuck_queue.py` | Manages "stuck problems" with bidirectional chaining |
| `src/problems/__init__.py` | Package init |
| `src/memory/causation_graph.py` | Personal causation graph built from all chains over time |
| `docs/deep-thought-build-guide.md` | This document |

### Existing Files to Modify

| File | Changes |
|---|---|
| `src/association_engine/tree_generator.py` | Add `DEEP_THOUGHT_HOP_TEMPLATE`, support configurable depth/pruning mode, add inefficiency scoring for Deep Thought pruning |
| `src/scoring/interest_scorer.py` | Add `score_collision()` method, new `CollisionScore` weights, add `mechanism_specificity` and `testability` LLM prompts |
| `src/bridge_builder/builder.py` | Add `build_hypothesis()` method, `HYPOTHESIS_SYSTEM` and `HYPOTHESIS_TEMPLATE` prompts, new `deep_thought` excitement tier |
| `src/config/settings.py` | Add `DeepThoughtConfig`, `StuckProblemConfig` dataclasses, `creativity_mode` to `EngineConfig`, update `load_config` |
| `config.example.yaml` | Add `creativity_mode`, `deep_thought` section, `stuck_problems` section |
| `src/models.py` | Add `CollisionResult`, `CollisionScore` dataclasses |
| `src/memory/store.py` | Add `find_cross_temporal_collisions()`, `store_intermediate_nodes()`, `get_all_intermediate_embeddings()` methods |
| `src/memory/incubation.py` | Add cross-temporal collision checks to hourly cycle |
| `src/main.py` | Add `--mode` CLI flag, `solve`/`problems`/`forget` commands, wire `MultiSeedGenerator`, `CollisionEngine`, `StuckQueue` into heartbeat cycle |

---

## 13. Prioritized Build Order

### Sprint 1: The Foundation (Days 1-3) 🔨

**Focus:** Multi-seed generation + Deep Thought prompts + config mode

1. Add `creativity_mode` and `DeepThoughtConfig` to `settings.py` and `config.example.yaml`
2. Add `DEEP_THOUGHT_HOP_TEMPLATE` to `tree_generator.py`
3. Add configurable depth/pruning mode to `tree_generator.py` (honor Deep Thought config)
4. Create `src/association_engine/multi_seed.py` (parallel chain generation from multiple seeds)
5. Wire into `main.py` with `--mode deep_thought` flag
6. **Test:** Run with a concrete example ("kitchen remodel + LED project") and compare output to normal mode

**Deliverable:** The engine generates 5 parallel chains per heartbeat from different seed sources. Each chain goes 15 hops deep with Deep Thought prompts. Output is still single-chain based (no collisions yet) but chains are wilder and longer.

### Sprint 2: The Collision Engine (Days 4-7) 💥

**Focus:** Bisociation detection between parallel chains

1. Add `CollisionResult` and `CollisionScore` to `models.py`
2. Create `src/association_engine/collision_engine.py`
3. Store all intermediate node embeddings during chain generation
4. Implement pairwise cosine similarity across chains from different seeds
5. Add `score_collision()` to `interest_scorer.py`
6. Wire collision results into the heartbeat cycle in `main.py`
7. **Test:** Generate parallel chains from "kitchen" + "cancer research" seeds and verify collisions are found at "blue light/circadian" type intersection points

**Deliverable:** The engine detects bisociation points between independent chains and surfaces them. First "magical" outputs should appear.

### Sprint 3: Hypothesis Output (Days 8-10) 🔮

**Focus:** Shift output from "fun facts" to "testable hypotheses"

1. Add `build_hypothesis()` to `bridge_builder/builder.py`
2. Add `HYPOTHESIS_SYSTEM` and `HYPOTHESIS_TEMPLATE`
3. Add `deep_thought` excitement tier
4. Format output with claim/chain/test/confidence structure
5. Add hypothesis-specific transparency reveal (show both chains + collision point)
6. **Test:** End-to-end: multi-seed → collision → hypothesis → transparency reveal

**Deliverable:** Deep Thought outputs look and feel like oracle pronouncements with testable claims.

### Sprint 4: Stuck Problems + Bidirectional Chaining (Days 11-16) 🎯

**Focus:** Let the user define problems; engine chains backward from targets

1. Create `src/problems/stuck_queue.py`
2. Add bidirectional chain generation (backward from target)
3. Store backward chains in ChromaDB with `direction: "backward"` metadata
4. Add collision detection between forward (ambient) chains and backward (problem) chains
5. Add `solve`, `problems`, `forget` commands to `main.py`
6. **Test:** Add "cure colon cancer" as a stuck problem. Over multiple heartbeats with kitchen/LED context, verify that forward-backward collisions are detected.

**Deliverable:** The engine works on your stuck problems in the background, surfacing hypotheses when your daily life accidentally collides with a solution path. This is the Harman/Fadiman 1966 study, deterministically.

### Sprint 5: Cross-Temporal Memory Collisions (Days 17-21) ⏳

**Focus:** Detect when today's chain collides with something from weeks ago

1. Add `store_intermediate_nodes()` and `find_cross_temporal_collisions()` to `memory/store.py`
2. Every chain stored: persist ALL intermediate node embeddings (not just endpoints)
3. On new chain generation, query memory for old intermediate nodes with high cosine similarity
4. Add to incubation cycle: cross-temporal collision sweep
5. **Test:** Simulate time-gap collisions by storing chains from different contexts, then generating new chains that should collide

**Deliverable:** The engine has "long-term memory collisions" — connecting something from 3 weeks ago to today's context through a hidden shared concept. This is the "7.5 million years" computation.

### Sprint 6: Personal Causation Graph (Days 22-28) 🕸️

**Focus:** Build a persistent graph of the user's entire causal universe

1. Create `src/memory/causation_graph.py`
2. Every intermediate node from every chain becomes a graph node
3. Adjacent chain nodes become graph edges
4. User ratings propagate affinity through graph
5. Graph queries: shortest path between concepts, hub nodes, personal domain map
6. Feed graph data into multi-seed selection (seeds chosen from graph periphery for maximum collision potential)
7. **Test:** After accumulating 50+ chains, query the graph for surprising shortest paths

**Deliverable:** The engine knows your entire personal concept universe and can find paths through it that you didn't know existed. This is genuine "personalized superintelligence."

---

## 14. Epistemological Safeguards

How we distinguish genuine hidden causation from LLM hallucination:

### 1. Grounded Intermediate Hops

Every hop in a Deep Thought chain gets a quick Tavily check. Each node is tagged:
- **Web-grounded** — A specific fact/paper/mechanism was found supporting this hop
- **LLM-inferred** — The LLM generated this association; no web confirmation yet
- **Speculative** — The hop is plausible but has no external support

The ratio of grounded to speculative hops determines confidence level.

### 2. Mechanism Requirement

Collisions must include a **specific causal mechanism** — not just "these concepts are related." The LLM is prompted: "What is the specific physical/chemical/biological/social MECHANISM by which A causes B at this collision point?" If it can't name one, the collision is downgraded.

### 3. Testable Predictions

Every hypothesis must include a testable prediction: "If this hidden link is real, then you should observe X when you do Y." Untestable hypotheses get flagged as "speculative / philosophical" rather than "actionable."

### 4. Confidence Levels

```
HIGH    — 70%+ of hops are web-grounded, mechanism is specific and published
MEDIUM  — 40-70% grounded, mechanism is plausible but not directly studied  
LOW     — <40% grounded, mechanism is speculative, but the pattern is real
ORACLE  — The chain is too long/complex to fully verify, but collision strength
          is very high — this is a "42" moment: the answer feels right even if
          you can't trace the full proof
```

### 5. Transparent Chain Reveal

The transparency toggle we already built becomes even more important in Deep Thought mode. When the user says `reveal`, they see BOTH chains, the collision point, the confidence level, and which hops are grounded vs. speculative. They can then decide whether to trust the hypothesis.

### 6. Hit Rate Tracking

The analytics module (`src/memory/analytics.py`) already tracks AHA! rate. Extend it to track:
- **Hypothesis verification rate** — Of hypotheses the user investigated, how many held up?
- **Collision accuracy** — Do high-confidence collisions correlate with high user ratings?
- **Grounding-to-insight ratio** — What % grounding produces the best insights?

Over time, this data tunes the engine's confidence calibration.

---

## Closing Philosophy

> *"The answer to the Ultimate Question of Life, the Universe, and Everything is... 42."*  
> — Deep Thought

The number sounds absurd because the Question is unknown. But the computation was 7.5 million years of genuine processing. The engine we're building does the same thing: it runs background computations (parallel chains, collision detection, cross-temporal memory sweeps) that are too complex for a human to follow. The output — a hypothesis — may sound as absurd as "42" or as magical as "paint your cabinet blue to affect colon cancer."

The causation is real. The chain is deterministic. The complexity is what makes it invisible to humans. And that invisibility is what makes it feel like superintelligence.

**If the causation is complex enough, it will seem like magic.**

That's the whole point. That's always been the whole point.

---

*This guide is a living document. Update it as each phase is completed.*
