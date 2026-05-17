# 🚀 Creativity Engine — Future Ideas

Ideas that are exciting but not yet the right priority. Revisit when the core engine is stable and the memory system has real data.

---

## 🖥️ Desktop Widget / Notifications

**Status:** Deferred — terminal is better for development iteration

**The Vision:**
Get the "tap on the shoulder" out of the terminal and into a physical desktop presence.

**Implementation Ideas:**
- **Phase 1: Native Notifications**
  - Use `plyer` for cross-platform desktop notifications
  - Notification includes the interjection text + action buttons
  - "Not now" button maps to `heartbeat.backoff()`
  - "Tell me more" button triggers co-creation mode
  
- **Phase 2: Transparent Overlay Widget**
  - Always-on-top transparent window (PyQt6 or tkinter)
  - Shows current status: heartbeat countdown, personality emoji, tempo
  - Interjections slide in as toast notifications
  - "Chain peek" toggle for power users — shows the association path
  - Dismissible with a gesture or keystroke
  
- **Phase 3: System Tray Integration**
  - Engine runs as a system tray app
  - Tray icon pulses when the engine is "thinking"
  - Right-click menu: Status, Force Fire, Mute, Quit
  - Double-click opens the overlay widget

**Key Libraries:**
- `plyer` — cross-platform notifications
- `pystray` — system tray integration
- `PyQt6` or `tkinter` — overlay widget
- `pyttsx3` — offline TTS fallback (no API cost)

**When to Build:**
When you stop wanting to see debug output (scoring breakdowns, chain summaries, novelty levels) and just want the interjections delivered naturally. Probably after 2-3 weeks of stable operation.

**Watch Out For:**
- Don't lose the debug visibility — keep terminal mode as a `--debug` flag
- The overlay needs to be truly non-intrusive — think macOS notification center, not modal dialog
- Multiple monitor support matters for gaming setups

---

## 🎨 Image Generation Output

**Status:** Deferred — cool but adds latency and cost

**The Vision:**
Some insights are inherently visual. When the engine connects "astronaut nutrition" to "bioluminescent deep-sea organisms," a generated image of glowing space food could make the interjection 10x more memorable.

**Implementation Ideas:**
- **Selective Generation Only**
  - Don't generate images for every interjection — most don't benefit
  - Only trigger for "mind_blown" tier (score > 0.85) where the endpoint is visual
  - Use a quick LLM check: "Would this insight benefit from a visual?" (yes/no)
  
- **API Options:**
  - OpenAI DALL-E 3 — highest quality, $0.04-0.08 per image
  - Grok Imagine (xAI) — fast, good quality
  - Stable Diffusion (local) — free but requires GPU and setup
  
- **Integration Points:**
  - Add `image_url` field to `Interjection` model
  - Bridge builder generates an image prompt alongside the text
  - Image displays in the desktop widget (when built) or opens in default viewer
  - Store in memory with the chain for later recall
  
- **Personality-Driven Style:**
  - Curious Nerd → clean scientific illustration style
  - Playful Provocateur → surreal, Dalí-esque style
  - Quiet Philosopher → minimalist, contemplative imagery
  - Excited Storyteller → dynamic, cinematic scenes
  - Pattern Spotter → abstract pattern/connection visualizations

**Cost Estimate:**
At ~10 interjections/day with 20% qualifying for images:
- DALL-E 3: ~$0.08-0.16/day ($2.40-4.80/month)
- Local SD: Free (but needs 6GB+ VRAM)

**When to Build:**
After the desktop widget exists (images need somewhere to display). Also after the scoring system is tuned enough that "mind_blown" tier truly means something special — you don't want to burn API credits on mediocre insights.

---

## 🚶 Mobile / Walk Mode

**Status:** Deferred — start with laptop walk mode using existing features

**The Vision:**
Creativity loves movement. The engine should be able to come with you on walks, where your changing environment becomes a constant stream of fresh input. Steve Jobs walked. Nietzsche walked. Your best ideas probably happen in the shower or on a walk — this mode would capture that.

**Phase 1: Laptop Walk Mode (near-term)**
Use what we already have — just in a different context:
- **Existing mic input** becomes ambient sound capture while walking (birds, traffic, music)
- **Screen context** shows whatever's on screen (podcast app, music player, map)
- **Shorter heartbeat interval** (1-2 min) because input changes faster when moving
- **Audio-only output** via TTS — you're walking, not reading a screen
- **Emotion/prosody detection** picks up your energy level from ambient sounds and responses
- **Config preset:** `python -m src.main --live --walk` activates walk-optimized settings:
  - Fast heartbeat (1-2 min cycles)
  - Voice output forced ON
  - Higher novelty sensitivity
  - Audio input sensitivity boosted

**Phase 2: Phone Companion (far-future)**
Full mobile experience using the phone's sensors:
- **Phone mic** for ambient audio and voice interaction
- **Phone camera** for visual context (what you're seeing on your walk)
- **GPS/location** for geo-aware seeds ("You're near the waterfront → water-based associations")
- **Step counter / movement detection** for activity-aware heartbeat pacing
- **Push notifications** for interjections when the phone is locked

**Implementation Ideas (Phase 2):**
- Lightweight phone app (React Native or Flutter) that streams sensor data to the engine
- Engine runs on laptop/home server, phone is a thin client
- WebSocket connection between phone and engine
- Alternatively: fully local phone version using a smaller LLM (Phi-3, Gemma)

**Key Libraries (Phase 1 — already have most of these):**
- Existing `AudioChannel` — already handles mic input
- Existing `VoiceOutput` — TTS for hands-free output
- Existing `ScreenChannel` — captures whatever's on screen
- New: CLI preset flag (`--walk`) that configures optimal walk settings

**Key Libraries (Phase 2 — future):**
- React Native / Flutter for mobile app
- WebSocket server (FastAPI/websockets) for real-time phone→engine streaming
- `geopy` for location-based context enrichment

**When to Build:**
- **Phase 1:** Anytime! It's mostly a config preset. Add it when you want to try a walking session.
- **Phase 2:** After Phase 1 proves walking sessions are genuinely useful. Don't build a phone app for a workflow you haven't validated with a laptop first.

**Watch Out For:**
- Battery drain on phone (constant mic + camera + GPS is heavy)
- Privacy: outdoor camera captures other people — need clear opt-in and no cloud storage
- Audio quality: wind noise, traffic, etc. — may need noise gate before processing
- Walking pace: interjections need to be SHORT in walk mode — can't process paragraphs while walking

**The Philosophy:**
This connects to the core thesis: creativity comes from the collision of ideas across contexts. Walking literally changes your context every few minutes. The engine + movement = maximum collision potential. 🚶‍♂️💡

---

## 📝 Notes

- Both desktop widget and image generation build on each other — that's the natural unlock point
- Image generation could also be a co-creation feature: "show me what that looks like"
- Consider ElevenLabs for premium voice variety (different from OpenAI TTS voices)
- The overlay widget could eventually show the incubation queue as a "thoughts percolating" animation
- Walk mode Phase 1 could be built THIS WEEK — it's mostly just a CLI preset
- Walk mode proves the thesis that changing physical context accelerates creative associations
