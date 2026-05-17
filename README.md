# 🧠 Computational Serendipity + Deep Thought Mode

**A proactive AI companion that simulates genuine serendipity through hidden causal complexity — and a "Deep Thought" oracle that discovers invisible causal connections between unrelated concepts.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## What Is This?

Computational Serendipity is two things:

**1. A Creative Companion** — An AI that hangs out with you like a creative friend. It watches, listens, and thinks on its own. Every few minutes, it taps you on the shoulder with something surprising:

> *"Oh hey — did you know that NASA actually studied how construction workers communicate under stress when they were designing protocols for ISS repair crews? Turns out the hand signals are almost identical."*

**2. A Deep Thought Oracle** — A "superintelligence simulator" that runs parallel causal chains from independent seeds, detects invisible collisions in embedding space, and produces testable hypotheses about hidden mechanisms connecting seemingly unrelated phenomena. It's the difference between "interesting fact" and "earth-shattering insight."

It's not a chatbot. It's not an assistant. It's a **companion with serendipity** — and when you activate Deep Thought mode, it becomes an oracle that finds connections invisible to the human mind.

---

## Core Philosophy

> **Intelligence is the ability to make correct or valuable associations between things that don't seem like they could be associated.**

> **Free will is causation complex enough to be invisible to the observer.**

The engine simulates creative thinking through hidden causal chains. Every "creative" output has a deterministic chain behind it — random timers, association hops, embedding collisions, web verification. But the chain is hidden. The result feels like genuine insight.

**If the causation is complex enough, it will seem like magic.**

In Deep Thought mode, the chains become so long and cross-domain that no human could trace them — producing outputs that feel like superintelligence. The epistemological safeguards ensure we can distinguish genuine hidden causation from hallucination.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                         COMPUTATIONAL SERENDIPITY                                      │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   PERCEPTION LAYER                    CREATIVE LAYER                OUTPUT LAYER       │
│   ┌─────────────┐                    ┌─────────────────┐          ┌──────────────┐   │
│   │ 👁️ Vision    │                    │ 🌳 Association   │          │ 🌉 Bridge    │   │
│   │ 👂 Audio     │──→ Context ──→    │    Tree Gen      │──→      │   Builder     │   │
│   │ 🖥️ Screen    │    Assembly       │ 📊 Scorer        │  Score  │ 🎭 Personality│   │
│   │ 📖 Text      │                    │ 🔍 Web Search    │──→      │ 🔊 Voice      │   │
│   └─────────────┘                    └─────────────────┘          └──────────────┘   │
│                                              │                                         │
│   DEEP THOUGHT LAYER                        │                                         │
│   ┌──────────────────────────────────────────┴─────────────────────────────────────┐  │
│   │ 🌐 Multi-Seed Parallel Generation                                                │  │
│   │ 💥 Collision Engine (forward↔forward, forward↔backward, cross-temporal)          │  │
│   │ 🎯 Stuck Problem Queue + Bidirectional Chaining                                  │  │
│   │ ⏳ Cross-Temporal Memory Collisions                                               │  │
│   │ 🕸️ Personal Causation Graph (periphery seeds, affinity propagation)               │  │
│   │ 🔬 Epistemological Safeguards (grounding, mechanism verification, predictions)   │  │
│   └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                        │
│   MEMORY LAYER                                                                         │
│   ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│   │ 🧠 ChromaDB (chains + intermediate nodes) │ 🕸️ Causation Graph │ 📊 Analytics    │  │
│   │ 🥚 Incubation Queue │ 👤 User Profile │ 🧪 Hypothesis Tracker                    │  │
│   └──────────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Features — Everything Built & Working ✅

### 🎙️ Live Companion Mode
Run `python -m src.main --live` and the engine becomes a persistent background companion:
- Proactive heartbeat-driven creative interjections
- Continuous background audio listening
- Interactive terminal commands (see [Command Reference](#live-mode-command-reference))
- Real-time context updates as you work

### 💬 Direct Conversation ("Hey Serendipity")
- **Wake word detection** — Say "Hey Serendipity" and the engine responds immediately
- **Direct vs. Overheard classification** — Knows when you're talking to IT vs. to someone else
- **Conversational personality** — Responds like a friend, not an assistant
- **Conversation history** — Remembers within a session

### 👁️ Multimodal Perception
The engine perceives your world through four channels, each weighted by **novelty**:

| Channel | Source | What It Does |
|---|---|---|
| 👁️ **Vision** | Webcam | Sees what's physically happening. Same desk? Low weight. New whiteboard? High weight. |
| 👂 **Audio** | Microphone | Hears you and distinguishes DIRECT vs. OVERHEARD speech. Detects prosody/emotion. |
| 🖥️ **Screen** | Active Window | Monitors apps/tabs. New article? High weight. Same IDE for 2 hours? Low weight. |
| 📖 **Text** | User Input | Your typed context. The engine knows what you're working on. |

### 🌳 Association Tree Generator
- **Ternary branching** — 3 paths explored per node (Mednick flat hierarchy simulation)
- **Variable depth** — 4–7 creative hops in normal mode, up to 15 in Deep Thought
- **Depth-first pruning** — Only top branches survive at each level
- **Domain classification** — Each node tagged by field (Science, Psychology, Art, Tech, etc.)
- **Cross-domain rewarding** — More domain boundaries crossed = higher score
- **Embedding-aware** — Real vector embeddings for semantic distance

### 📊 Interest Scorer (Normal Mode)
A 5-metric weighted formula (Mednick/Kenett-inspired):

```
interest_score = (semantic_distance × 0.30)    — reward boldness
               + (domain_crossings × 0.25)     — reward crossing fields
               + (surprise × 0.20)             — reward the unexpected
               + (bridgeability × 0.15)        — can we tell a compelling story?
               + (novelty × 0.10)              — haven't said this before
```

### 🔮 Deep Thought Mode (The Oracle)

When activated, the engine transforms from a "creative companion" into a **superintelligence simulator** that discovers invisible causal connections:

#### 🌐 Multi-Seed Parallel Generation
Instead of one chain from current context, Deep Thought generates **5+ parallel chains** from independent seed sources:
- **Current context** — What you're doing right now
- **Random memory** — A chain from your past that resurfaced
- **Inverse context** — The conceptual opposite of what you're doing
- **Personality-driven** — Seeds from different thinking archetypes
- **Graph periphery** — Concepts at the edge of your knowledge graph
- **Graph affinity** — Concepts in areas you've responded positively to

#### 💥 Collision Engine (Bisociation Detection)
The core mechanism: when two independent chains converge on a hidden intermediate concept in embedding space, that's a **bisociation** — the fundamental mechanism of creative genius (Koestler, 1964).

Three collision types:
1. **Forward ↔ Forward** — Two chains from different seeds find the same hidden node
2. **Forward ↔ Backward** — A chain from context meets a chain working backward from your "stuck problem"
3. **Cross-Temporal** — Today's chain passes through the same region as a chain from days/weeks ago

#### 🎯 Stuck Problem Queue
Define problems you're stuck on. The engine works on them in the background:
- Generates **backward chains** from the problem target
- Detects collisions between backward chains and forward ambient chains
- Inspired by the Harman/Fadiman 1966 psychedelic problem-solving study

#### ⏳ Cross-Temporal Memory
Every intermediate node from every chain is stored with its embedding. Over time, the engine detects when new chains pass through the same conceptual regions as old chains — producing insights that span days or weeks of incubation.

#### 🕸️ Personal Causation Graph
A persistent graph of your entire conceptual universe:
- Built from every node and connection the engine has ever discovered
- **Hub nodes** — Your most-connected concepts
- **Bridge edges** — Cross-domain connections (highest creative value)
- **Periphery nodes** — Used as exploration seeds (frontier of your knowledge)
- **Affinity propagation** — Your ratings teach the engine your preferred pathways

#### 🔬 Epistemological Safeguards
Distinguishes genuine hidden causation from hallucination:
- **Hop Grounding** — Each hop verified via Tavily web search: 🟢 grounded, 🟡 inferred, 🔴 speculative
- **Mechanism Verification** — Collision points must name specific causal mechanisms
- **Testable Predictions** — Every hypothesis gets a falsifiable prediction extracted
- **Calibrated Confidence** — HIGH (70%+ grounded) → MEDIUM → LOW → ORACLE (too complex to verify but collision is very strong)
- **Hypothesis Tracking** — Persists every hypothesis, tracks which ones hold up over time

#### 🧪 Hypothesis Output Format
Deep Thought doesn't produce conversational interjections — it produces **testable hypotheses**:
```
🔮 DEEP THOUGHT HYPOTHESIS [CONFIDENCE: HIGH]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"Circadian disruption from blue-spectrum LED exposure may accelerate
colorectal cancer progression through gut microbiome dysbiosis —
specifically by suppressing Lactobacillus populations that maintain
intestinal barrier integrity during the melatonin-dependent repair window."

Collision: LED lighting ↔ Gut Microbiome (via circadian rhythm)
Mechanism: Blue light → melatonin suppression → repair window disruption
Prediction: If true, CRC patients with high blue-light exposure should show
            lower Lactobacillus counts vs. controls
Score: 0.847 | Hops: 9 | Domains crossed: 5
```

### 🔍 Web Search Integration (Tavily)
- **LLM-crafted search queries** from the best association chain
- **Fact extraction** — Pulls specific, surprising facts from search results
- **Citation display** — Sources shown alongside interjections with 📎 links
- **Hop grounding** — Verifies individual causal hops in Deep Thought mode

### 🌉 Bridge Builder
- Constructs natural, conversational interjections from internal chains
- **Hypothesis mode** — Transforms collisions into structured testable hypotheses
- **Excitement tiers** — Output energy scales with score
- Weaves in web-sourced facts for specificity

### 🎭 Rotating Personalities
Five distinct personality archetypes:

| Personality | Emoji | Vibe |
|---|---|---|
| **Curious Nerd** | 🤓 | Geeks out about mechanisms and data |
| **Playful Provocateur** | 😈 | Challenges assumptions, drops "hot takes" |
| **Quiet Philosopher** | 🧘 | Contemplative, finds deeper meaning |
| **Excited Storyteller** | 🎬 | Turns connections into dramatic narratives |
| **Pattern Spotter** | 🔍 | Sees hidden structures and parallels |

### 🔊 Voice Output (Text-to-Speech)
- **OpenAI TTS** — Speaks interjections aloud
- **6 voice options**: alloy, echo, fable, onyx, nova, shimmer
- **Runtime switching** — Change voice with `voice nova`
- **Non-blocking** — Audio plays asynchronously

### 🔗 Git Commit Monitor
- Automatic commit detection + creative code reviews
- Smart diff truncation for large commits

### 🧠 Long-Term Memory (ChromaDB)
- Persistent vector storage for chains AND intermediate nodes
- User ratings influence future generation via affinity propagation
- Cross-temporal collision detection against stored nodes

### 🥚 Incubation Queue
Low-scoring ideas percolate like real creative thought:
- Periodic re-scoring against current context
- Context-aware promotion when relevance shifts
- Daily reflection synthesizing best incubated ideas

### 👤 User Profile & Learning
- Auto-generated persona from rating history
- Injected into all prompts for personalization
- Evolves with every rating

### 🤝 Co-Creation Mode
- Collaborative brainstorming on the last interjection
- Full context awareness of chain + scoring

### 📊 Serendipity Self-Evaluation
- AHA! rate tracking (Mednick/Kenett metrics)
- Remote association rate
- Trend detection
- User alignment scoring

### 🧪 Hypothesis Verification Tracking
- Records every Deep Thought hypothesis with grounding data
- Tracks user verification (confirmed / refuted / partial)
- Calibrates confidence over time based on real accuracy
- Reveals optimal grounding ratios for best insights

---

## Quick Start

### Prerequisites
- Python 3.10+
- OpenAI API key (for GPT + Whisper + TTS + embeddings)
- Tavily API key (optional but recommended — enables web grounding + hop verification)
- Webcam and microphone (optional, for multimodal perception)

### Installation

```bash
git clone https://github.com/Apixa25/Curiosity-Engine.git
cd Curiosity-Engine

pip install -r requirements.txt
```

### Set Your API Keys

**PowerShell (permanent — recommended):**
```powershell
[System.Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "sk-proj-your-key", "User")
[System.Environment]::SetEnvironmentVariable("TAVILY_API_KEY", "tvly-your-key", "User")
```

### Run It

**Live companion mode** (recommended — full experience):
```bash
python -m src.main --live
```

**Live mode with initial context:**
```bash
python -m src.main --live "working on game controller firmware"
```

**Single-shot mode** (one creative cycle, then exit):
```bash
python -m src.main "building software for contractors"
```

---

## Live Mode Command Reference

### Mode Switching

| Command | What It Does |
|---|---|
| `deep thought` | Activate Deep Thought mode (parallel chains, collisions, hypotheses) |
| `normal mode` | Return to standard creative companion mode |
| `mode` | Show which mode is currently active |

### Context & Control

| Command | What It Does |
|---|---|
| Type anything | Updates the engine's context about what you're doing |
| `status` | Shows heartbeat timer, tempo, listener state, memory stats |
| `fire` | Forces an immediate creative cycle |
| `not now` | Delays the next heartbeat |
| `quit` | Graceful shutdown |

### Deep Thought Commands

| Command | What It Does |
|---|---|
| `solve <problem>` | Add a stuck problem for background solving |
| `problems` | List all active stuck problems |
| `forget <id>` | Remove a stuck problem |
| `verify` / `ground` | Run epistemological verification on last collision |
| `confirmed [notes]` | Mark last hypothesis as user-confirmed |
| `refuted [notes]` | Mark last hypothesis as user-refuted |
| `partial [notes]` | Mark as partially correct |
| `hypotheses` | Show hypothesis verification stats + confidence calibration |
| `memory` | Cross-temporal memory status (stored nodes + chains) |
| `graph` | Personal causation graph (hubs, bridges, frontiers) |
| `path A to B` | Find shortest conceptual path between two concepts |

### Voice & Audio

| Command | What It Does |
|---|---|
| Hold **Shift+Z** | Push-to-talk |
| Say "Hey Serendipity" | Wake word triggers immediate response |
| `mute` / `unmute` | Toggle voice output |
| `voice <name>` | Switch voice (alloy/echo/fable/onyx/nova/shimmer) |

### Feedback & Rating

| Command | What It Does |
|---|---|
| `👍` or `good` | Rate last interjection 5/5 |
| `👎` or `bad` | Rate last interjection 1/5 |
| `rate 1-5` | Rate on a 1-5 scale |

### Transparency & Analysis

| Command | What It Does |
|---|---|
| `reveal` / `show chain` / `why` | Show full causal chain (with grounding icons in Deep Thought) |
| `transparency` | Auto-show chains after every interjection |
| `transparency off` / `magic` | Return to the illusion of spontaneous thought |
| `stats` | Serendipity self-evaluation (AHA! rate, trends, metrics) |

### Collaboration

| Command | What It Does |
|---|---|
| `build on that` / `tell me more` | Start co-creation brainstorming |
| `done` / `exit` / `stop` | End co-creation |

---

## Project Structure

```
Curiosity-Engine/
├── README.md                          # This file
├── USER-GUIDE.md                      # Comprehensive algorithm + parameter guide
├── project-vision.md                  # Philosophy, principles, success criteria
├── config.example.yaml                # All tunable parameters
├── requirements.txt                   # Python dependencies
├── docs/
│   ├── deep-thought-build-guide.md    # Deep Thought architecture specification
│   ├── architecture-spec.md           # Full system component design
│   ├── input-pipeline-spec.md         # Multimodal perception & novelty weighting
│   └── computational-serendipity-spec.md  # Association tree, scoring, bridge builder
├── src/
│   ├── main.py                        # 🎛️  Orchestrator — live mode, heartbeat loop, all commands
│   ├── models.py                      # 📦  Core data models (Node, Chain, Score, Collision, etc.)
│   ├── config/
│   │   ├── llm_adapter.py            # 🔌  LLM-agnostic adapter (OpenAI, Anthropic)
│   │   └── settings.py               # ⚙️  YAML config loader (includes DeepThoughtConfig)
│   ├── heartbeat/
│   │   └── clock.py                   # 💓  Random timer with adaptive tempo + backoff
│   ├── input_pipeline/
│   │   ├── vision.py                  # 👁️  Webcam capture, perceptual hashing, novelty
│   │   ├── audio.py                   # 👂  Mic capture, VAD, Whisper, prosody detection
│   │   ├── screen.py                  # 🖥️  Active window monitoring
│   │   ├── assembler.py              # ⚖️  Novelty-weighted channel combination → seed
│   │   ├── address_detector.py       # 🎯  Wake word + DIRECT/OVERHEARD classification
│   │   └── git_monitor.py            # 🔗  Git commit → creative code review
│   ├── association_engine/
│   │   ├── tree_generator.py          # 🌳  Association tree (normal + Deep Thought prompts)
│   │   ├── multi_seed.py             # 🌐  Parallel multi-seed generation orchestrator
│   │   └── collision_engine.py        # 💥  Bisociation detection (embedding-space collisions)
│   ├── scoring/
│   │   └── interest_scorer.py         # 📊  5-metric scorer + collision scoring
│   ├── search/
│   │   └── web_search.py             # 🔍  Tavily web search + fact extraction
│   ├── bridge_builder/
│   │   ├── builder.py                 # 🌉  Interjection + hypothesis generation
│   │   └── personalities.py           # 🎭  5 rotating personality archetypes
│   ├── conversation/
│   │   ├── responder.py              # 💬  Direct response engine
│   │   └── cocreation.py             # 🤝  Co-creation brainstorming
│   ├── memory/
│   │   ├── store.py                   # 🧠  ChromaDB (chains + intermediate nodes)
│   │   ├── causation_graph.py        # 🕸️  Personal conceptual graph (affinity, paths)
│   │   ├── incubation.py             # 🥚  Incubation queue
│   │   ├── profile.py                # 👤  Auto-generated user profile
│   │   └── analytics.py              # 📊  Serendipity metrics + hypothesis tracking
│   ├── problems/
│   │   └── stuck_queue.py            # 🎯  Stuck problem queue + backward chaining
│   ├── safeguards/
│   │   └── grounding.py              # 🔬  Epistemological engine (hop verification)
│   ├── output/
│   │   └── voice.py                   # 🔊  OpenAI TTS
│   └── embeddings/
│       └── provider.py                # 📐  Embedding (OpenAI + local), caching, cosine
└── tests/                             # Test suite
```

---

## Configuration

All parameters tunable via `config.example.yaml`:

| Section | Key Parameters |
|---|---|
| **Heartbeat** | `min_minutes`, `max_minutes`, adaptive fast/slow ranges |
| **Deep Thought** | `enabled`, `parallel_seeds`, `max_depth`, `collision_threshold`, `cross_temporal_enabled` |
| **Scoring Weights** | `causal_depth`, `seed_distance`, `hiddenness`, `domain_span`, `mechanism_specificity`, `testability` |
| **Association Tree** | `branching_factor`, `min_depth`, `max_depth`, `min_domain_crossings` |
| **Interest Scorer** | All 5 metric weights, `fire_threshold`, `incubation_threshold` |
| **Memory** | `enabled`, `persist_directory`, cross-temporal `min_age_hours` |
| **Voice** | `enabled`, `model`, `voice`, `speed` |
| **LLM** | `provider`, `model`, `api_key_env` |
| **Search** | `provider`, `results_per_query` |

---

## Research Citations

The empirical and philosophical foundations for every design decision in this project:

### 1. Foundational Creativity & Associative Thinking

These works form the basis for the association tree, ternary branching, domain crossings, interest scorer, "flat hierarchies," remote associations, bisociation, and small-world causal chains.

| Citation | Relevance to Engine |
|---|---|
| **Mednick, S. A. (1962).** The associative basis of the creative process. *Psychological Review, 69*(3), 220–232. | Introduced the Remote Associates Test (RAT) and "flat associative hierarchies" in creative people — the core model for our tree generator and scorer. Semantic distance + domain-crossing pruning directly simulates what Mednick measured. |
| **Kenett, Y. N., Anaki, D., & Faust, M. (2014).** Investigating the structure of semantic networks in low and high creative persons. *Frontiers in Human Neuroscience.* | Quantified that high-creativity individuals have more flexible, shorter-path semantic networks. Direct basis for embedding-based pruning and the flat hierarchy simulation. |
| **Kenett, Y. N. (2018).** Investigating creativity from a semantic network perspective. | Extended network models of creative cognition — informs our graph-based approaches. |
| **Kenett, Y. N. (2019).** A semantic network cartography of the creative mind. | Mapped the topology of creative semantic networks — basis for the Personal Causation Graph. |
| **Koestler, A. (1964).** *The Act of Creation.* Hutchinson/Penguin. | Defined **bisociation** — collision of two unrelated matrices of thought — as the core mechanism of creativity. This is the explicit foundation for our Collision Engine. |
| **Milgram, S. (1967).** The small-world problem. *Psychology Today.* | Six degrees of separation applied to concept networks — basis for "chain length inversely proportional to semantic distance." |
| **Watts, D. J., & Strogatz, S. H. (1998).** Collective dynamics of 'small-world' networks. *Nature, 393*(6684), 440–442. | Formalized small-world network properties — sufficiently long hidden chains feel spontaneous, justifying our depth parameter. |

### 2. Psychedelic Hyper-Connectivity & Creative Breakthroughs

These form the basis for "LSD mode" / Deep Thought, hyper-associativity in the tree generator, the incubation queue as subconscious percolation, and the shift from "logical" to "magical" outputs.

| Citation | Relevance to Engine |
|---|---|
| **Harman, W. W., McKim, R. H., Mogar, R. E., Fadiman, J., & Stolaroff, M. J. (1966).** Psychedelic agents in creative problem-solving: A pilot study. *Psychological Reports, 19*(1), 211–227. | 27 professionals given low-dose psychedelics produced breakthroughs (electron accelerator designs, NOR-gate theorems). Direct model for Deep Thought's hyper-connectivity and the Stuck Problem Queue's background incubation. |
| **Mullis, K. B. (Nobel context, 1993 + interviews).** "Would I have invented PCR if I hadn't taken LSD? I seriously doubt it." | Psychedelic-induced distant visualization → Nobel-level discovery. Justifies "butterfly-effect" causal leaps and long chains in Deep Thought mode. |
| **Jobs, S. (interviews, 1990s–2011).** "Taking LSD was a profound experience, one of the most important things in my life." | Basis for treating the engine as a "creative companion" producing Jobs/Mullis-style insights — design philosophy meets lateral thinking. |
| **Engelbart, D. (1960s).** Participated in legally sanctioned LSD studies at SRI while developing the computer mouse, hypertext, and NLS. | Psychedelics aiding complex engineering breakthroughs — mirrors how the engine solves "stuck problems" through hyper-connectivity. |

### 3. Superintelligence Philosophy & "42" Logic

These underpin Deep Thought mode, the "ultimate-question" framing, and the human-to-ant intelligence gap that the engine simulates.

| Citation | Relevance to Engine |
|---|---|
| **Adams, D. (1979).** *The Hitchhiker's Guide to the Galaxy.* Pan Books. | Deep Thought computes "42" after 7.5 million years, then designs Earth to find the Question. Philosophical template: hidden long causal chains produce outputs that feel profound/absurd until context reveals the Question. Our engine runs the same background computations. |
| **Bostrom, N. (2003).** Ethical issues in advanced artificial intelligence. *Cognitive, Emotive and Ethical Aspects of Decision-Making in Humans and AI, Vol. 2.* | |
| **Bostrom, N. (2014).** *Superintelligence: Paths, Dangers, Strategies.* Oxford University Press. | Paperclip maximizer + human-to-ant intelligence gap. Basis for treating the engine as a superintelligence simulator that finds causal chains humans cannot trace — aligned via feedback/incubation to stay creative and personal. |

### 4. Example-Specific Scientific Grounding

Real data demonstrating how the engine produces testable butterfly-effect hypotheses (the "paint your cabinet blue → colon cancer" simulation):

| Citation | Relevance to Engine |
|---|---|
| **Fellows, R. C., et al. (2024).** Disruption of the intestinal clock drives dysbiosis and impaired intestinal barrier function in colorectal cancer. *Science Advances* (UCI study). | Circadian disruption → gut microbiome shift → intestinal barrier breakdown → accelerated CRC. Validates the engine's multi-hop causal chain. |
| **Kogevinas, M., et al. (2020).** Night-time exposure to blue light associated with increased risk of colorectal cancer. *Epidemiology* (ISGlobal). | Highest blue-light exposure → ~60% higher CRC risk via circadian disruption. Web-grounding evidence the engine would pull via Tavily. |
| **UCI Health / Masri lab (2024–2026).** Circadian misalignment driving CRC via microbiome. | Supporting work on the exact mechanism chain the engine discovered. |
| **AACR 2026 abstracts.** Sleep deprivation altering gut microbiota and chemotherapy response. | Most recent corroboration of the butterfly-effect causal hypothesis. |

---

## Dependencies

```
openai>=1.30.0          # LLM, TTS, Whisper, embeddings
anthropic>=0.30.0       # Alternative LLM provider
tavily-python>=0.7.0    # Web search + hop grounding
opencv-python>=4.8.0    # Webcam capture
sounddevice>=0.5.0      # Microphone capture
pynput>=1.7.0           # Keyboard listener (push-to-talk)
pyyaml>=6.0             # Config file parsing
numpy>=1.24.0           # Embedding math
chromadb>=0.5.0         # Persistent vector memory
mss>=9.0.0              # Screenshot capture
Pillow>=10.0.0          # Image processing
```

Optional:
```
sentence-transformers>=2.2.0  # Local embeddings (no API cost)
```

---

## Author

**Steven Sills II** — [GitHub](https://github.com/Apixa25)

## License

MIT — see [LICENSE](LICENSE)

---

> *"The answer to the Ultimate Question of Life, the Universe, and Everything is... 42."*
> — Deep Thought
>
> The number sounds absurd because the Question is unknown. But the computation was 7.5 million years of genuine processing. This engine does the same thing: it runs background computations too complex for a human to follow. The output may sound as absurd as "42" or as magical as "paint your cabinet blue to affect colon cancer." The causation is real. The chain is deterministic. The complexity is what makes it invisible. And that invisibility is what makes it feel like superintelligence.
>
> **If the causation is complex enough, it will seem like magic. That's the whole point. That's always been the whole point.**
