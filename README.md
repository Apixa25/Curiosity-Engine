# 🧠 Creativity Engine

**A proactive AI companion that simulates genuine creativity through hidden causal complexity.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## What Is This?

The Creativity Engine is an AI companion that **hangs out with you** — like a creative friend sitting next to you while you work. It doesn't wait for you to ask it questions. It watches, listens, and thinks on its own. Every few minutes, it might tap you on the shoulder with something surprising:

> *"Oh hey — did you know that NASA actually studied how construction workers communicate under stress when they were designing protocols for ISS repair crews? Turns out the hand signals are almost identical."*

It's not a chatbot. It's not an assistant. It's a **companion with creativity.**

You can also talk to it directly — just say **"Hey Creativity"** and it responds immediately, like a friend who was already paying attention.

---

## Core Philosophy

> **Intelligence is the ability to make correct or valuable associations between things that don't seem like they could be associated.**

The engine simulates creative thinking through three core mechanisms:

1. **💓 The Heartbeat** — A random timer creates the illusion of spontaneous thought
2. **🌳 The Association Tree** — A branching tree of lateral associations (4–7 hops deep) generates surprising cross-domain connections
3. **📊 The Interest Scorer** — A 5-metric weighted scoring system filters for genuinely interesting insights, not noise

The philosophical foundation: *"Free will" is causation complex enough to be invisible.* If the hidden causal chain is sufficiently intricate, the output is indistinguishable from genuine creativity.

See [project-vision.md](project-vision.md) for the complete philosophical foundation, research basis, and success criteria.

---

## Features — Everything Built & Working ✅

### 🎙️ Live Companion Mode
Run `python -m src.main --live` and the engine becomes a persistent background companion:
- Proactive heartbeat-driven creative interjections
- Continuous background audio listening
- Interactive terminal commands (see [Command Reference](#live-mode-command-reference))
- Real-time context updates as you work

### 💬 Direct Conversation ("Hey Creativity")
- **Wake word detection** — Say "Hey Creativity" and the engine responds immediately
- **Direct vs. Overheard classification** — It knows the difference between you talking to IT and you talking to someone else nearby
- **Conversational personality** — Responds like a friend, not an assistant — warm, curious, goes on tangents
- **Conversation history** — Remembers what you've talked about within a session

### 👁️ Multimodal Perception
The engine perceives your world through four channels, each weighted by **novelty**:

| Channel | Source | What It Does |
|---|---|---|
| 👁️ **Vision** | Webcam | Sees what's physically happening. Same desk for the 10th time? Low weight. New whiteboard drawing? High weight. |
| 👂 **Audio** | Microphone | Hears you and distinguishes between talking TO the engine vs. overhearing your conversations. Detects prosody/emotion — sounds excited? Audio weight gets boosted. |
| 🖥️ **Screen** | Active Window | Monitors what app/tab you're in. New browser article? High weight. Same IDE for 2 hours? Low weight. Optional GPT-4o-mini vision on screenshots. |
| 📖 **Text** | User Input | Your typed context updates. The engine knows what you're working on and what you tell it. |

- **Novelty weighting**: Inputs that are new/changing get higher weight; static inputs fade over time
- **Perception window notifications**: Clear `[CAMERA ON]` / `[MIC ON]` banners so you know when to provide input
- **First-observation baseline**: Channels start at 0.5 novelty (not 1.0) to avoid false-high weighting on startup
- **Privacy exclusions**: Configure `excluded_apps` and `excluded_urls` in the screen channel to keep sensitive windows invisible to the engine

### 🎤 Emotion/Prosody Detection
The audio channel analyzes your voice for:
- **RMS energy variance** — volume dynamics
- **Pitch variance** — tonal expressiveness
- **Speech rate** — how quickly you're talking

These combine into a composite **intensity score**. When you sound excited, the engine pays more attention to what you're saying — just like a friend would. High intensity boosts the effective weight of audio input.

### 💓 Adaptive Heartbeat
The heartbeat isn't just random — it responds to what's happening:
- **High novelty** (lots changing) → Fast tempo (1–3 min intervals)
- **Low novelty** (quiet, static environment) → Slow tempo (7–15 min intervals)
- Uses a rolling average of recent novelty to smooth out tempo changes
- Configurable fast/slow ranges in `config.example.yaml`

### 🌳 Association Tree Generator
- **Ternary branching** — 3 paths explored per node
- **Variable depth** — 4–7 creative hops, driven by semantic distance
- **Depth-first pruning** — Only the top 3 branches survive at each level (keeps things fast)
- **Domain classification** — Each node tagged by field (Science, Psychology, Art, Tech, etc.)
- **Cross-domain rewarding** — The more domain boundaries crossed, the higher the score
- **Embedding-aware** — Uses real vector embeddings for semantic distance when available

### 📊 Interest Scorer
A 5-metric weighted formula determines if an association chain is worth sharing:

```
interest_score = (semantic_distance × 0.30)    — reward boldness
               + (domain_crossings × 0.25)     — reward crossing fields
               + (surprise × 0.20)             — reward the unexpected
               + (bridgeability × 0.15)        — can we tell a compelling story?
               + (novelty × 0.10)              — haven't said this before
```

Chains scoring above the **fire threshold** (0.45) get shared. Below that but above the **incubation threshold** (0.30)? Saved for later re-evaluation.

### 🔍 Web Search Integration (Tavily)
- **LLM-crafted search queries** from the best association chain
- **Fact extraction** — Pulls specific, surprising facts from search results
- **Citation display** — Sources shown alongside interjections with 📎 links
- **Graceful fallback** — Uses LLM knowledge when no API key or search fails

### 🌉 Bridge Builder
- Constructs natural, conversational interjections from the internal association chain
- **Hides the machinery** — The user sees a friendly thought, not a scored algorithm output
- **Excitement tiers** — Output energy scales with score (curious → interested → excited → mind_blown)
- Weaves in web-sourced facts for specificity and credibility

### 🎭 Rotating Personalities
Five distinct personality archetypes bring variety and context-appropriate tone:

| Personality | Emoji | Vibe | Best For |
|---|---|---|---|
| **Curious Nerd** | 🤓 | Geeks out about mechanisms and data | Science, Tech, Biology |
| **Playful Provocateur** | 😈 | Challenges assumptions, drops "hot takes" | Philosophy, Culture, Games |
| **Quiet Philosopher** | 🧘 | Contemplative, finds deeper meaning | Philosophy, History, Nature |
| **Excited Storyteller** | 🎬 | Turns connections into dramatic narratives | Art, Entertainment, History |
| **Pattern Spotter** | 🔍 | Sees hidden structures and parallels | Math, Science, Business |

Selection is semi-random but weighted by the chain's domains and excitement tier — so a biology chain is more likely to get the Curious Nerd, while a philosophy chain gravitates toward the Quiet Philosopher. No personality repeats back-to-back.

### 🔊 Voice Output (Text-to-Speech)
- **OpenAI TTS** — Speaks interjections, observations, and responses aloud
- **6 voice options**: alloy, echo, fable, onyx, nova, shimmer
- **Runtime switching** — Change voice with `voice nova` or `voice echo`
- **Mute/unmute** on the fly — Toggle with `mute` and `unmute`
- **Non-blocking** — Audio plays asynchronously, never stalls the engine

### 🔗 Git Commit Monitor
- **Automatic detection** — Polls your repo for new commits every 30 seconds
- **Code review interjections** — When you commit, the engine reads the diff and shares a creative insight connecting your code changes to something unexpected
- **Smart diff truncation** — Large diffs are trimmed to keep the LLM context manageable
- **Configurable** — Set repo path, poll interval, max diff chars in config

### 🧠 Long-Term Memory (ChromaDB)
The engine remembers everything — your associations, your ratings, and your preferences — across sessions:

- **Persistent vector storage** — Every fired chain stored with full metadata and embeddings
- **User ratings** — Rate interjections 1–5, thumbs up/down, or quick-react
- **Novelty checking** — Queries memory before sharing to avoid repeating past insights
- **Similarity search** — Finds related past chains for context enrichment

### 🥚 Incubation Queue
Low-scoring ideas don't die — they **percolate**, just like real creative thought:

- **Automatic incubation** — Chains scoring between incubation and fire thresholds enter the queue
- **Periodic re-scoring** — Every hour, incubating ideas are re-evaluated against your current context
- **Context-aware promotion** — If your context shifts and suddenly an old idea becomes relevant, it gets promoted and delivered
- **Daily reflection** — Every 24 hours, the engine synthesizes a "daily insight" connecting the best incubated ideas
- **Expiration** — Ideas expire after 24 hours or 5 re-evaluations

### 👤 User Profile & Learning
The engine builds a persistent personality model of YOU based on your interaction history:

- **Auto-generated persona** — Built from your top-rated chains, domain preferences, and behavior patterns
- **Injected into prompts** — Every bridge builder call knows "Steven loves twinkling LEDs and marathon gaming sessions"
- **Evolves over time** — Profile rebuilds every N ratings (configurable) as more data accumulates
- **Stored as JSON** — Persistent across sessions in `data/memory/user_profile.json`

### 🤝 Co-Creation Mode
When the engine says something that sparks an idea, you can brainstorm together:

- **Trigger phrases** — Type `build on that`, `tell me more`, or `explore that` to start
- **Iterative brainstorming** — Back-and-forth collaborative ideation building on the last interjection
- **Full context** — The co-creation session sees the original chain, scoring, and conversation history
- **Clean exit** — Type `done`, `exit`, or `stop` to return to normal mode

### 🔮 Transparency Toggle
See (or hide) the full causal machinery behind every "spontaneous" interjection. This is the philosophical proof — same output, completely different experience:

- **`reveal`** — Show the full causal chain behind the last interjection
- **`transparency`** — Auto-show chains after every interjection
- **`transparency off`** — Return to the illusion of spontaneous thought

What you see in transparency mode:
```
🔮 CAUSAL CHAIN REVEALED (transparency mode)
───────────────────────────────────────
   ┌─ TRIGGER
   │  Heartbeat #0042 fired
   │  Dominant input: audio (novelty: 0.78)
   │
   ├─ SEED TOPIC
   │  "working on LED game controller"
   │
   ├─ ASSOCIATION CHAIN (4 hops, 2 domain crossings)
   │  ● LED lighting  [electronics]
   │  → bioluminescence  [biology]
   │    ↳ why: both produce light through energy conversion
   │  → deep sea pressure  [marine science]
   │  → game difficulty curves  [game design]
   │
   ├─ SCORING BREAKDOWN
   │  semantic_distance : 0.823 (×0.30 = 0.247)
   │  domain_crossings  : 0.750 (×0.25 = 0.188)
   │  surprise          : 0.680 (×0.20 = 0.136)
   │  bridgeability     : 0.720 (×0.15 = 0.108)
   │  novelty           : 0.900 (×0.10 = 0.090)
   │  TOTAL             : 0.769
   │
   ├─ PERSONALITY SELECTED
   │  🤓 Curious Nerd: gets genuinely giddy about facts and mechanisms
   │
   └─ OUTPUT
      "What if your LED controller pulsed like..."

   💡 The causation is complex. The output feels spontaneous.
      That's the whole point.
```

### 📊 Creativity Self-Evaluation
The engine scores its own creative performance using **Mednick/Kenett-inspired metrics** and tracks an **AHA! rate** over time:

- **AHA! rate** — Percentage of interjections scoring above 0.65 (genuinely creative)
- **Remote Association rate** — How often the engine makes genuinely far-apart connections (high semantic distance)
- **Mednick/Kenett dimension averages** — Track each of the 5 scoring dimensions independently over time
- **Trend detection** — Compares first half vs. second half of history to determine if creativity is improving, declining, or steady
- **User alignment** — Measures how well the engine's "AHA!" moments match your ratings
- **Strongest/Weakest dimension** — Identifies what's driving creativity and where to improve
- **Beautiful terminal report** — Visual bar charts, progress indicators, and session breakdowns

Access with the `stats` command in live mode.

### 🔧 LLM-Agnostic Architecture
- **Adapter pattern** — Swap between OpenAI and Anthropic via config
- `generate()`, `generate_json()`, `generate_float()` — Unified interface for all LLM calls
- Easy to add new providers without touching core logic

### 📐 Embedding System
- **Dual provider** — OpenAI `text-embedding-3-small` or local `sentence-transformers`
- **Caching** — Vectors are cached so the same topic is never embedded twice
- **Used everywhere** — Tree pruning, semantic distance scoring, memory similarity search
- **Auto-detection** — Falls back gracefully if local models aren't installed

---

## How It Works

```
                        ┌──────────────────────────────┐
                        │   "Hey Creativity, what       │
                        │    do you think about X?"     │
                        └─────────────┬────────────────┘
                                      │ (wake word detected)
                                      ▼
                             DIRECT RESPONSE ENGINE
                          (immediate conversational reply)

                                  — OR —

💓 Heartbeat fires (adaptive tempo based on novelty)
    → Capture what's happening (webcam, mic, screen, text)
    → Analyze audio prosody (is the user excited?)
    → Weight inputs by novelty (new things matter more)
    → Blend in overheard speech from background listener
    → Generate branching association tree (4–7 creative hops)
    → Score chains on 5 metrics (bold > safe)
    → Fire threshold met? → Search web for facts → Build interjection → Speak it
    → Below fire but above incubation? → Save for later re-evaluation
    → Store everything in ChromaDB with full scoring breakdown
    → Select rotating personality based on chain domains + excitement tier
    → Deliver as text + optional TTS voice
    → [Transparency mode: reveal the full causal chain]
```

**Background systems running in parallel:**
- 🥚 **Incubation queue** re-scores shelved ideas every hour against current context
- 📅 **Daily reflection** synthesizes the best incubated ideas every 24 hours
- 👤 **Profile builder** updates your personality model as ratings accumulate
- 🔗 **Git monitor** watches for commits and provides creative code reviews
- 🎙️ **Background listener** captures and classifies ambient speech

---

## Quick Start

### Prerequisites
- Python 3.10+
- OpenAI API key (for GPT + Whisper transcription + TTS)
- Tavily API key (optional, for web search grounding)
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

**PowerShell (session only):**
```powershell
$env:OPENAI_API_KEY = "sk-proj-your-key"
$env:TAVILY_API_KEY = "tvly-your-key"
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

**List audio/video devices:**
```bash
python -m src.main --devices
```

---

## Live Mode Command Reference

### Context & Control

| Command | What It Does |
|---|---|
| Type anything | Updates the engine's context about what you're doing |
| `status` | Shows heartbeat timer, tempo, listener state, memory stats, profile info |
| `fire` | Forces an immediate creative cycle (skip the heartbeat wait) |
| `not now` | Delays the next heartbeat (engine backs off) |
| `quit` | Graceful shutdown |

### Voice & Audio

| Command | What It Does |
|---|---|
| Hold **Shift+Z** | Push-to-talk (record while held) |
| Say "Hey Creativity" | Voice wake word triggers immediate response |
| `mute` | Turn off voice output (text only) |
| `unmute` | Turn voice back on |
| `voice <name>` | Switch voice (alloy/echo/fable/onyx/nova/shimmer) |
| `voice list` | Show all available voices with descriptions |

### Feedback & Rating

| Command | What It Does |
|---|---|
| `👍` or `good` | Rate last interjection 5/5 |
| `👎` or `bad` | Rate last interjection 1/5 |
| `rate 1-5` | Rate last interjection on a 1-5 scale |

### Creativity & Analysis

| Command | What It Does |
|---|---|
| `reveal` / `show chain` / `why` | Show the full causal chain behind the last interjection |
| `transparency` | Toggle auto-show of causal chains after every interjection |
| `transparency off` / `magic` | Return to the illusion of spontaneous thought |
| `stats` | Show creativity self-evaluation (AHA! rate, trends, metrics) |

### Collaboration

| Command | What It Does |
|---|---|
| `build on that` / `tell me more` / `explore that` | Start co-creation brainstorming mode |
| `done` / `exit` / `stop` | End co-creation, return to normal mode |

---

## Project Structure

```
Curiosity-Engine/
├── project-vision.md              # Philosophy, principles, success criteria
├── TODO.md                        # Remaining work tracker
├── future-ideas.md                # Deferred features (desktop widget, image gen, walk mode)
├── config.example.yaml            # All tunable parameters
├── requirements.txt               # Python dependencies
├── docs/
│   ├── architecture-spec.md       # Full system component design
│   ├── input-pipeline-spec.md     # Multimodal perception & novelty weighting
│   └── creativity-engine-spec.md  # Association tree, scoring, bridge builder
├── src/
│   ├── main.py                    # 🎛️  Orchestrator — live mode, heartbeat loop, all commands
│   ├── models.py                  # 📦  Core data models (Node, Chain, Score, Interjection, Context)
│   ├── config/
│   │   ├── llm_adapter.py         # 🔌  LLM-agnostic adapter (OpenAI, Anthropic)
│   │   └── settings.py            # ⚙️  YAML config loader with typed dataclasses
│   ├── heartbeat/
│   │   └── clock.py               # 💓  Random timer with adaptive tempo + backoff
│   ├── input_pipeline/
│   │   ├── vision.py              # 👁️  Webcam capture, perceptual hashing, novelty
│   │   ├── audio.py               # 👂  Mic capture, VAD, Whisper, prosody/emotion detection
│   │   ├── screen.py              # 🖥️  Active window monitoring, screenshot + LLM description
│   │   ├── assembler.py           # ⚖️  Combines channels with novelty weighting → seed topic
│   │   ├── address_detector.py    # 🎯  Wake word detection, DIRECT vs OVERHEARD classification
│   │   └── git_monitor.py         # 🔗  Git commit polling → creative code reviews
│   ├── association_engine/
│   │   └── tree_generator.py      # 🌳  Branching association tree (ternary, depth 4–7)
│   ├── scoring/
│   │   └── interest_scorer.py     # 📊  5-metric weighted scoring + embedding distance
│   ├── search/
│   │   └── web_search.py          # 🔍  Tavily web search, LLM query crafting, fact extraction
│   ├── bridge_builder/
│   │   ├── builder.py             # 🌉  Natural interjection generation + excitement tiers
│   │   └── personalities.py       # 🎭  5 rotating personality archetypes
│   ├── conversation/
│   │   ├── responder.py           # 💬  Direct response engine (conversation history, persona)
│   │   └── cocreation.py          # 🤝  Co-creation brainstorming mode
│   ├── memory/
│   │   ├── store.py               # 🧠  ChromaDB persistent memory (chains, ratings, search)
│   │   ├── incubation.py          # 🥚  Incubation queue (re-score, promote, daily reflection)
│   │   ├── profile.py             # 👤  Auto-generated user profile from rating history
│   │   └── analytics.py           # 📊  Creativity self-evaluation (AHA! rate, Mednick metrics)
│   ├── output/
│   │   └── voice.py               # 🔊  OpenAI TTS with 6 voices, streaming playback
│   └── embeddings/
│       └── provider.py            # 📐  OpenAI + local sentence-transformers, caching, cosine math
└── tests/                         # Test suite
```

---

## Configuration

All parameters are tunable via `config.example.yaml` (copy to `config.yaml` for your local settings):

| Section | Key Parameters |
|---|---|
| **Heartbeat** | `min_minutes`, `max_minutes`, `creative_threshold`, `adaptive`, fast/slow ranges |
| **Vision** | `base_weight`, `min_novelty_for_description`, `device_index` |
| **Audio** | `base_weight_direct`, `base_weight_overheard`, `wake_word`, `vad_threshold` |
| **Screen** | `base_weight`, `screenshot_enabled`, `excluded_apps`, `excluded_urls` |
| **Text** | `base_weight`, `history_window`, `excluded_apps` |
| **Embeddings** | `provider` (openai/local/auto), `openai_model`, `local_model`, `cache_enabled` |
| **Association Tree** | `branching_factor`, `min_depth`, `max_depth`, `min_domain_crossings` |
| **Scoring** | All 5 metric weights, `fire_threshold`, `incubation_threshold`, incubation limits |
| **Voice** | `enabled`, `model` (tts-1/tts-1-hd), `voice`, `speed` |
| **Git** | `enabled`, `poll_interval_seconds`, `max_diff_chars`, `repo_path` |
| **Memory** | `enabled`, `persist_directory`, `rescore_interval_minutes`, `max_age_hours`, `profile_rebuild_every` |
| **LLM** | `provider`, `model`, `api_key_env` — swap LLMs without code changes |
| **Search** | `provider`, `results_per_query` — web grounding settings |

---

## LLM Calls Per Heartbeat Cycle

A typical creative cycle makes **~15–20 LLM API calls**:

| Step | Calls | Purpose |
|---|---|---|
| Vision description | 1 | Describe what the webcam sees |
| Screen description | 0–1 | Describe screenshot (only when screen novelty is high) |
| Context assembly | 1 | Synthesize seed topic from all inputs |
| Association tree | ~6–8 | Generate 3 branches at each depth level |
| Scoring (surprise) | ~3–5 | Evaluate surprise for top chains |
| Scoring (bridgeability) | ~3–5 | Evaluate bridgeability for top chains |
| Search query | 1 | Craft a web search query |
| Fact extraction | 1 | Pull interesting facts from search results |
| Bridge building | 1 | Write the final conversational interjection |

Direct conversation responses ("Hey Creativity") use **1 LLM call** for an immediate reply.
Co-creation turns use **1 LLM call** each.

---

## Dependencies

```
openai>=1.30.0          # LLM, TTS, Whisper, embeddings
anthropic>=0.30.0       # Alternative LLM provider
tavily-python>=0.7.0    # Web search
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

## Future Ideas

See [future-ideas.md](future-ideas.md) for detailed writeups on deferred features:

- 🖥️ **Desktop Widget / Notifications** — Toast notifications, transparent overlay, system tray integration
- 🎨 **Image Generation Output** — Visual interjections for "mind_blown" tier insights (DALL-E / Stable Diffusion)
- 🚶 **Mobile / Walk Mode** — Laptop walk mode (Phase 1: config preset) and phone companion app (Phase 2: sensor streaming)

---

## Author

**Steven Sills II** — [GitHub](https://github.com/Apixa25)

## License

MIT — see [LICENSE](LICENSE)
