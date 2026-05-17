"""
Computational Serendipity — Proof of Concept Runner

Three modes:
    python -m src.main                          # interactive (manual fire)
    python -m src.main "topic"                  # single-fire with seed topic
    python -m src.main --live                   # LIVE companion mode (continuous heartbeat)
    python -m src.main --live "working on code" # live mode with initial context
"""

from __future__ import annotations

import asyncio
import sys
import time

from src.config.settings import load_config, EngineConfig
from src.config.llm_adapter import create_llm_adapter, LLMAdapter
from src.heartbeat.clock import Heartbeat
from src.association_engine.tree_generator import AssociationTreeGenerator
from src.scoring.interest_scorer import InterestScorer
from src.bridge_builder.builder import BridgeBuilder
from src.search.web_search import WebSearcher
from src.input_pipeline.vision import VisionChannel
from src.input_pipeline.audio import AudioChannel
from src.input_pipeline.screen import ScreenChannel
from src.input_pipeline.assembler import ContextAssembler
from src.input_pipeline.address_detector import AddressDetector
from src.conversation.responder import DirectResponder
from src.conversation.cocreation import CoCreator
from src.output.voice import VoiceOutput, VoiceConfig
from src.input_pipeline.git_monitor import GitMonitor
from src.embeddings.provider import EmbeddingProvider, EmbeddingConfig
from src.memory.store import MemoryStore, MemoryConfig
from src.memory.incubation import IncubationQueue
from src.memory.profile import ProfileBuilder
from src.memory.analytics import SerendipityAnalytics
from src.association_engine.multi_seed import MultiSeedGenerator
from src.models import ContextSnapshot, Interjection


class ComputationalSerendipity:
    """Orchestrates the full creative pipeline."""

    def __init__(self, config: EngineConfig | None = None, debug_audio: bool = False):
        self.cfg = config or load_config()
        self.debug_audio = debug_audio
        self.llm: LLMAdapter = create_llm_adapter(
            provider=self.cfg.llm.provider,
            model=self.cfg.llm.model,
            api_key_env=self.cfg.llm.api_key_env,
        )
        self.heartbeat = Heartbeat(
            min_minutes=self.cfg.heartbeat.min_minutes,
            max_minutes=self.cfg.heartbeat.max_minutes,
            adaptive=self.cfg.heartbeat.adaptive,
            fast_min=self.cfg.heartbeat.fast_min,
            fast_max=self.cfg.heartbeat.fast_max,
            slow_min=self.cfg.heartbeat.slow_min,
            slow_max=self.cfg.heartbeat.slow_max,
        )
        self.embedder = EmbeddingProvider(EmbeddingConfig(
            provider=self.cfg.embeddings.provider,
            openai_model=self.cfg.embeddings.openai_model,
            local_model=self.cfg.embeddings.local_model,
            cache_enabled=self.cfg.embeddings.cache_enabled,
        ))
        self.tree_gen = AssociationTreeGenerator(self.llm, self.cfg.association_tree, self.embedder)
        self.scorer = InterestScorer(self.llm, self.cfg.scoring, self.embedder)
        self.bridge = BridgeBuilder(self.llm, persona="")
        self.searcher = WebSearcher(self.llm)
        self.past_topics: list[str] = []
        self.current_context: str = ""
        self._thinking = False
        self._listening = False
        self._multimodal = False
        self._force_creative = False
        self.vision: VisionChannel | None = None
        self.audio: AudioChannel | None = None
        self.screen: ScreenChannel | None = None
        self.assembler: ContextAssembler | None = None
        self.detector: AddressDetector = AddressDetector(llm=self.llm)
        self.responder: DirectResponder = DirectResponder(llm=self.llm)
        self.cocreator: CoCreator = CoCreator(llm=self.llm)
        self._last_interjection: Interjection | None = None
        self._transparency = False
        self.voice: VoiceOutput = VoiceOutput(VoiceConfig(
            enabled=self.cfg.voice.enabled,
            model=self.cfg.voice.model,
            voice=self.cfg.voice.voice,
            speed=self.cfg.voice.speed,
        ))
        self.git_monitor: GitMonitor = GitMonitor(
            repo_path=self.cfg.git.repo_path,
            poll_interval_seconds=self.cfg.git.poll_interval_seconds,
            max_diff_chars=self.cfg.git.max_diff_chars,
        )
        self.memory = MemoryStore(MemoryConfig(
            enabled=self.cfg.memory.enabled,
            persist_directory=self.cfg.memory.persist_directory,
            collection_name=self.cfg.memory.collection_name,
        ))
        self.incubation = IncubationQueue(
            memory=self.memory,
            llm=self.llm,
            rescore_interval_minutes=self.cfg.memory.rescore_interval_minutes,
            max_age_hours=self.cfg.memory.max_age_hours,
            max_rescores=self.cfg.memory.max_rescores,
        )
        self.profile = ProfileBuilder(
            memory=self.memory,
            llm=self.llm,
            profile_path=self.cfg.memory.persist_directory + "/user_profile.json",
            rebuild_every_n_ratings=self.cfg.memory.profile_rebuild_every,
        )
        self.analytics = SerendipityAnalytics(memory=self.memory)
        self._overheard_buffer: list[str] = []
        self._deep_thought_active = self.cfg.creativity_mode == "deep_thought" or self.cfg.deep_thought.enabled
        self.multi_seed: MultiSeedGenerator | None = None
        if self._deep_thought_active:
            self.multi_seed = MultiSeedGenerator(
                llm=self.llm,
                tree_config=self.cfg.association_tree,
                dt_config=self.cfg.deep_thought,
                embedder=self.embedder,
                memory=self.memory,
            )

    def enable_multimodal(self) -> None:
        """Initialize vision + audio channels. Call before run_live for full perception."""
        print("\n   Initializing multimodal input pipeline...")
        ip = self.cfg.input_pipeline

        self.vision = VisionChannel(
            history_window=ip.vision.history_window,
            base_weight=ip.vision.base_weight,
            min_novelty_for_description=ip.vision.min_novelty_for_description,
            device_index=ip.vision.device_index,
        )
        if ip.vision.enabled:
            self.vision.initialize()

        self.audio = AudioChannel(
            api_key=getattr(self.llm, 'api_key', ''),
            capture_seconds=ip.audio.capture_seconds,
            base_weight_direct=ip.audio.base_weight_direct,
            base_weight_overheard=ip.audio.base_weight_overheard,
            device_index=ip.audio.device_index,
            vad_threshold=ip.audio.vad_threshold,
        )
        if ip.audio.enabled:
            self.audio.initialize()

        self.screen = ScreenChannel(
            base_weight=ip.screen.base_weight,
            history_window=ip.screen.history_window,
            screenshot_enabled=ip.screen.screenshot_enabled,
            min_novelty_for_screenshot=ip.screen.min_novelty_for_screenshot,
            excluded_apps=ip.screen.excluded_apps,
            excluded_urls=ip.screen.excluded_urls,
        )
        if ip.screen.enabled:
            self.screen.initialize()

        self.assembler = ContextAssembler(
            llm=self.llm,
            vision=self.vision if self.vision.is_available else None,
            audio=self.audio if self.audio.is_available else None,
            screen=self.screen if self.screen.is_available else None,
        )
        self._multimodal = (
            self.vision.is_available or self.audio.is_available or self.screen.is_available
        )
        if self._multimodal:
            print("   Multimodal input enabled!")
        else:
            print("   No cameras or mics detected -- using text context only")

    async def run_creative_cycle(self, seed_topic: str, verbose: bool = True) -> Interjection | None:
        """
        Run one full creative cycle: tree → score → search → bridge → interjection.
        Returns the full Interjection object, or None if nothing interesting enough.
        """
        self._thinking = True
        t0 = time.time()

        try:
            ctx = ContextSnapshot(seed_topic=seed_topic, heartbeat_id=f"hb-{self.heartbeat.beat_count:04d}")

            if verbose:
                print(f"\n   🌳 Generating association tree from: \"{seed_topic}\"")

            chains = await self.tree_gen.generate_tree(seed_topic)

            if not chains:
                if verbose:
                    print("   ❌ No chains generated.")
                return None

            if verbose:
                print(f"   📊 Scoring top {min(5, len(chains))} of {len(chains)} chains...")

            ranked = await self.scorer.rank_chains(chains, ctx, self.past_topics)
            best_chain, best_score = ranked[0]

            if verbose:
                print(f"   🏆 Best: {best_chain.endpoint_topic} (score: {best_score.total:.3f})")

            if best_score.total < self.cfg.scoring.fire_threshold:
                if verbose:
                    print(f"   🤫 Score {best_score.total:.3f} below threshold {self.cfg.scoring.fire_threshold}")
                self._incubate_chains(ranked, seed_topic)
                return None

            if verbose:
                print(f"   🔍 Searching web for facts...")

            search_result = None
            try:
                search_result = await self.searcher.search_for_chain(
                    endpoint_topic=best_chain.endpoint_topic,
                    chain_summary=best_chain.summary(),
                    context=seed_topic,
                )
                if verbose and search_result and search_result.facts:
                    print(f"   ✅ Found {len(search_result.facts)} facts")
            except Exception as e:
                if verbose:
                    print(f"   ⚠️  Search failed: {e}")

            interjection = await self.bridge.build_interjection(
                best_chain,
                best_score,
                ctx,
                search_facts=search_result.facts if search_result else None,
                search_sources=search_result.source_urls if search_result else None,
            )
            self.past_topics.append(best_chain.endpoint_topic)
            self.scorer.record_interjection(best_chain)

            self._store_fired_chain(best_chain, best_score, interjection, seed_topic)
            self._incubate_chains(ranked[1:], seed_topic)

            elapsed = time.time() - t0
            if verbose:
                print(f"   ⏱️  Cycle took {elapsed:.1f}s")

            return interjection

        finally:
            self._thinking = False

    async def run_deep_thought_cycle(self, seed_topic: str, verbose: bool = True) -> Interjection | None:
        """Run a Deep Thought cycle: multi-seed parallel chains → best chain → interjection.

        In Sprint 1, this generates parallel chains from multiple seeds and picks
        the single best chain (same as normal mode, but from a much wider search).
        Sprint 2 will add collision detection between chains from different seeds.
        """
        if not self.multi_seed:
            return await self.run_creative_cycle(seed_topic, verbose=verbose)

        self._thinking = True
        t0 = time.time()

        try:
            ctx = ContextSnapshot(seed_topic=seed_topic, heartbeat_id=f"hb-{self.heartbeat.beat_count:04d}")

            if verbose:
                print(f"\n   🔮 DEEP THOUGHT CYCLE from: \"{seed_topic[:60]}\"")

            result = await self.multi_seed.generate_parallel(
                current_context=seed_topic,
                heartbeat_id=ctx.heartbeat_id,
            )

            if not result.all_chains:
                if verbose:
                    print("   ❌ No chains generated across any seed.")
                return None

            if verbose:
                print(f"   📊 Scoring top {min(5, len(result.all_chains))} of {result.total_chains} chains from {result.seed_count} seeds...")

            ranked = await self.scorer.rank_chains(result.all_chains, ctx, self.past_topics)
            best_chain, best_score = ranked[0]

            seed_label = "unknown"
            for label, chains in result.seed_chains.items():
                if best_chain in chains:
                    seed_label = label
                    break

            if verbose:
                print(f"   🏆 Best: {best_chain.endpoint_topic} (score: {best_score.total:.3f}, from seed: {seed_label})")
                print(f"   🔗 Chain length: {len(best_chain.nodes)} hops, {best_chain.domain_crossings} domain crossings")

            if best_score.total < self.cfg.scoring.fire_threshold:
                if verbose:
                    print(f"   🤫 Score {best_score.total:.3f} below threshold {self.cfg.scoring.fire_threshold}")
                self._incubate_chains(ranked, seed_topic)
                return None

            if verbose:
                print(f"   🔍 Searching web for facts to ground the chain...")

            search_result = None
            try:
                search_result = await self.searcher.search_for_chain(
                    endpoint_topic=best_chain.endpoint_topic,
                    chain_summary=best_chain.summary(),
                    context=seed_topic,
                )
                if verbose and search_result and search_result.facts:
                    print(f"   ✅ Found {len(search_result.facts)} grounding facts")
            except Exception as e:
                if verbose:
                    print(f"   ⚠️  Search failed: {e}")

            interjection = await self.bridge.build_interjection(
                best_chain,
                best_score,
                ctx,
                search_facts=search_result.facts if search_result else None,
                search_sources=search_result.source_urls if search_result else None,
            )
            self.past_topics.append(best_chain.endpoint_topic)
            self.scorer.record_interjection(best_chain)

            self._store_fired_chain(best_chain, best_score, interjection, seed_topic)
            self._incubate_chains(ranked[1:], seed_topic)

            elapsed = time.time() - t0
            if verbose:
                print(f"   ⏱️  Deep Thought cycle took {elapsed:.1f}s")

            return interjection

        finally:
            self._thinking = False

    def _store_fired_chain(self, chain, score, interjection, seed_topic: str) -> None:
        """Persist the fired chain in long-term memory with full scoring breakdown."""
        if not self.memory.is_available:
            return

        import uuid
        endpoint = chain.nodes[-1] if chain.nodes else None
        embedding = endpoint.embedding if endpoint else None

        self.memory.store_chain(
            chain_id=f"chain-{uuid.uuid4().hex[:12]}",
            seed_topic=seed_topic,
            endpoint_topic=chain.endpoint_topic,
            chain_summary=chain.summary(),
            domains=chain.nodes[-1].chain_domains() if chain.nodes else [],
            domain_crossings=chain.domain_crossings,
            score=score.total,
            embedding=embedding,
            interjection_text=interjection.interjection_text if interjection else "",
            context=self.current_context,
            status="fired",
            score_semantic_distance=score.semantic_distance,
            score_domain_crossings=score.domain_crossings,
            score_surprise=score.surprise,
            score_bridgeability=score.bridgeability,
            score_novelty=score.novelty,
        )

    def _incubate_chains(self, ranked_chains, seed_topic: str) -> None:
        """Send promising-but-not-ready chains to the incubation queue."""
        if not self.memory.is_available:
            return

        import uuid
        threshold = self.cfg.scoring.incubation_threshold

        for chain, score in ranked_chains:
            if threshold <= score.total < self.cfg.scoring.fire_threshold:
                endpoint = chain.nodes[-1] if chain.nodes else None
                embedding = endpoint.embedding if endpoint else None
                self.incubation.incubate(
                    chain_id=f"incub-{uuid.uuid4().hex[:12]}",
                    seed_topic=seed_topic,
                    endpoint_topic=chain.endpoint_topic,
                    chain_summary=chain.summary(),
                    domains=chain.nodes[-1].chain_domains() if chain.nodes else [],
                    domain_crossings=chain.domain_crossings,
                    score=score.total,
                    embedding=embedding,
                    context=self.current_context,
                )

    async def _on_incubation_promotion(self, stored_chain, new_score: float) -> None:
        """Called when an incubating chain ripens and should be delivered."""
        if stored_chain.interjection_text:
            text = stored_chain.interjection_text
        else:
            text = (f"You know what just clicked? Earlier I was thinking about "
                    f"\"{stored_chain.seed_topic}\" and how it connects to "
                    f"\"{stored_chain.endpoint_topic}\" — and now with what you're "
                    f"doing, that connection feels so much more relevant.")

        print(f"\n{'═' * 70}")
        print(f"🧪 INCUBATED IDEA HATCHED  [score: {new_score:.2f}]:\n")
        print(f"   \"{text}\"")
        print(f"\n{'═' * 70}")
        self.responder.add_engine_interjection(text)
        await self._speak(text)

    # ── LIVE COMPANION MODE ──────────────────────────────────────────

    async def run_live(self, initial_context: str = "") -> None:
        """
        Live companion mode — the engine runs continuously in the background.
        The heartbeat fires on its own timer. The user can update context,
        talk to the engine, or tell it to back off — all while it thinks.
        """
        self.current_context = initial_context

        print("=" * 70)
        print("🧠 COMPUTATIONAL SERENDIPITY — Live Companion Mode")
        print("=" * 70)

        import os
        api_key = getattr(self.llm, 'api_key', '') or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("\n   ⚠️  WARNING: No OPENAI_API_KEY found!")
            print("   Set it with:  $env:OPENAI_API_KEY = \"sk-proj-your-key-here\"")
            print("   The engine will NOT work without it.\n")

        print(f"\n   Provider: {self.cfg.llm.provider} | Model: {self.cfg.llm.model}")
        search_status = "✅ Tavily" if self.searcher.is_available else "⚠️  LLM fallback"
        print(f"   Search: {search_status}")
        print(f"   Heartbeat: {self.cfg.heartbeat.min_minutes}–{self.cfg.heartbeat.max_minutes} min")
        if self.current_context:
            print(f"   Context: \"{self.current_context}\"")

        self.embedder.initialize()
        self.enable_multimodal()
        self.voice.initialize()
        self.memory.initialize()
        self.profile.initialize()
        if self.profile.has_profile:
            self.bridge.persona = self.profile.persona_injection
        if self.cfg.git.enabled:
            self.git_monitor.initialize()

        has_mic = self.audio and self.audio.is_available
        if has_mic:
            print(f"   Mic Input: Hold Shift+Z to talk directly!")
        else:
            print(f"   Mic Input: No mic -- type to chat")
        voice_status = f"🔊 {self.voice.cfg.voice}" if self.voice.is_available else "🔇 Disabled"
        print(f"   Voice Output: {voice_status}")
        if self._deep_thought_active:
            print(f"   🔮 DEEP THOUGHT MODE ACTIVE")
            print(f"      Parallel seeds: {self.cfg.deep_thought.parallel_seeds} | Depth: {self.cfg.deep_thought.max_depth} | Min crossings: {self.cfg.deep_thought.min_domain_crossings}")

        print(f"\n{'─' * 70}")
        print("   Commands while running:")
        print("     Just type anything  → Update your context (what you're working on)")
        if has_mic:
            print("     Hold Shift+Z        → Push-to-talk (record while held)")
            print("     'Hey Serendipity'   → Voice wake word (also works)")
        print("     'not now'           → Skip next 2 heartbeats")
        print("     'status'            → Show engine status")
        print("     'fire'              → Force a heartbeat right now")
        print("     'mute' / 'unmute'   → Toggle voice output on/off")
        print("     'voice <name>'      → Switch voice (alloy/echo/fable/onyx/nova/shimmer)")
        print("     'voice list'        → Show all available voices")
        print("     '👍' or 'good'      → Rate last interjection positively")
        print("     '👎' or 'bad'       → Rate last interjection negatively")
        print("     'rate 1-5'          → Rate last interjection (1=terrible, 5=amazing)")
        print("     'build on that'     → Start brainstorming together (co-creation mode)")
        print("     'done'              → End brainstorm, back to normal")
        print("     'reveal'            → Show the full causal chain behind last interjection")
        print("     'transparency'      → Auto-show causal chains (toggle on/off)")
        print("     'stats'             → Serendipity metrics and AHA! rate over time")
        print("     'deep thought'      → Activate Deep Thought mode (parallel chains, butterfly-effect)")
        print("     'normal mode'       → Return to normal creative companion mode")
        print("     'mode'              → Show current creativity mode")
        print("     'quit'              → Shut down")
        print(f"{'─' * 70}")

        if not self.current_context:
            print("\n   💡 Tell me what you're up to! (or just press Enter to start)")
            try:
                initial = await asyncio.get_event_loop().run_in_executor(None, input, "   📝 Context: ")
                if initial.strip():
                    self.current_context = initial.strip()
                    print(f"   ✅ Got it — I'll hang out while you work on: \"{self.current_context}\"")
                else:
                    self.current_context = "general exploration"
                    print(f"   ✅ No worries — I'll just explore whatever catches my interest!")
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Shutting down.")
                return

        self._ptt_recording = False
        self._ptt_keys_held = set()
        self._ptt_audio_chunks = []

        tasks = [
            asyncio.create_task(self.heartbeat.start(self._on_heartbeat)),
            asyncio.create_task(self._input_loop()),
        ]

        if self.audio and self.audio.is_available:
            self._start_push_to_talk()
            tasks.append(asyncio.create_task(self._ptt_recording_loop()))
            tasks.append(asyncio.create_task(self._listening_loop()))

        if self.git_monitor.is_available:
            tasks.append(asyncio.create_task(self.git_monitor.start(self._on_new_commit)))

        if self.memory.is_available:
            tasks.append(asyncio.create_task(
                self.incubation.start(on_promotion=self._on_incubation_promotion)
            ))

        try:
            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        except asyncio.CancelledError:
            pass

        self._listening = False
        self.git_monitor.stop()
        self.incubation.stop()
        print("\n👋 Computational Serendipity shutting down. Stay creative!")
        if self.vision:
            self.vision.release()

    async def _on_heartbeat(self, ctx: ContextSnapshot) -> None:
        """Called by the heartbeat timer. Decides between observation mode
        (casual comment on what's happening) and creative mode (full
        association pipeline) based on the overall novelty of inputs."""
        if self.audio:
            self.audio.paused = True
            await asyncio.sleep(0.6)

        context_with_overheard = self.current_context
        if self._overheard_buffer:
            overheard_text = " | ".join(self._overheard_buffer[-3:])
            context_with_overheard += f" (overheard: {overheard_text})"
            self._overheard_buffer.clear()
        self.incubation.set_context(context_with_overheard)

        if self._multimodal and self.assembler:
            ctx = await self.assembler.assemble(
                user_text=context_with_overheard,
                heartbeat_id=ctx.heartbeat_id,
            )
            seed = ctx.seed_topic
        else:
            ctx.seed_topic = context_with_overheard
            seed = context_with_overheard
            print(f"   🎯 Context: \"{context_with_overheard}\"")

        threshold = self.cfg.heartbeat.creative_threshold
        go_creative = ctx.overall_novelty >= threshold or self._force_creative
        self._force_creative = False

        self.heartbeat.adjust_tempo(ctx.overall_novelty)

        if go_creative:
            print(f"   >> Novelty {ctx.overall_novelty:.2f} >= {threshold} — CREATIVE MODE")
            await self._heartbeat_creative(ctx, seed)
        else:
            print(f"   >> Novelty {ctx.overall_novelty:.2f} < {threshold} — OBSERVATION MODE")
            await self._heartbeat_observation(ctx)

        if self.audio:
            self.audio.paused = False

    async def _heartbeat_creative(self, ctx: ContextSnapshot, seed: str) -> None:
        """Full creative pipeline: association tree + scoring + search + bridge.
        In Deep Thought mode, routes through multi-seed parallel generation."""
        if self._deep_thought_active:
            interjection = await self.run_deep_thought_cycle(seed, verbose=True)
        else:
            interjection = await self.run_creative_cycle(seed, verbose=True)

        if interjection:
            self._last_interjection = interjection
            from src.bridge_builder.builder import _get_excitement_tier
            tier = _get_excitement_tier(interjection.scoring.total)
            tier_label = tier["name"].upper()
            personality_tag = f" | {self.bridge._last_personality.emoji} {self.bridge._last_personality.name}" if self.bridge._last_personality else ""
            print(f"\n{'═' * 70}")
            print(f"💬 SERENDIPITY SAYS  [{tier_label} | score: {interjection.scoring.total:.2f}{personality_tag}]:\n")
            print(f"   \"{interjection.interjection_text}\"")
            print(f"\n{'═' * 70}")
            self._print_citations(interjection)
            if self._transparency:
                self._print_causal_chain(interjection)
            self.responder.add_engine_interjection(interjection.interjection_text)
            await self._speak(interjection.interjection_text)
        else:
            print(f"\n   🤫 Nothing interesting enough this time. I'll keep thinking...")

    async def _heartbeat_observation(self, ctx: ContextSnapshot) -> None:
        """Lightweight observation: single LLM call commenting on what's happening.
        If the LLM has nothing to say, occasionally nudges the user instead."""
        self._thinking = True
        try:
            observation = await self.bridge.build_observation(ctx)
            if observation:
                print(f"\n{'─' * 50}")
                print(f"💭 SERENDIPITY OBSERVES  [novelty: {ctx.overall_novelty:.2f}]:\n")
                print(f"   \"{observation}\"")
                print(f"{'─' * 50}")
                self.responder.add_engine_interjection(observation)
                await self._speak(observation)
            else:
                print(f"\n   😌 Nothing to comment on — just vibing.")
        finally:
            self._thinking = False

    # ── GIT COMMIT REVIEW (direct insight, no creative associations) ──

    async def _on_new_commit(self, commit_info) -> None:
        """Called when the git monitor detects a new commit.
        Bypasses the entire creative pipeline — goes straight to the LLM
        for direct, thoughtful feedback on what was actually committed."""
        if self.audio:
            self.audio.paused = True

        self._thinking = True
        try:
            print(f"\n{'═' * 70}")
            print(f"📝 GIT COMMIT DETECTED")
            print(f"   {commit_info.hash_short} on {commit_info.branch}")
            print(f"   \"{commit_info.message.splitlines()[0][:70]}\"")
            print(f"   {commit_info.stats}")
            if commit_info.files_changed:
                for f in commit_info.files_changed[:8]:
                    print(f"      {f}")
                if len(commit_info.files_changed) > 8:
                    print(f"      ... and {len(commit_info.files_changed) - 8} more files")
            print(f"{'─' * 70}")
            print(f"   🧠 Reading the diff and thinking...")

            review = await self.bridge.build_commit_review(
                commit_info,
                user_context=self.current_context,
            )

            if review:
                print(f"\n{'═' * 70}")
                print(f"💬 SERENDIPITY ON YOUR COMMIT [{commit_info.hash_short}]:\n")
                print(f"   \"{review}\"")
                print(f"\n{'═' * 70}")
                self.responder.add_engine_interjection(review)
                await self._speak(review)
            else:
                print(f"   📝 Saw the commit, nothing specific to say about this one.")

        except Exception as e:
            print(f"   [Git Review] Error: {e}")
        finally:
            self._thinking = False
            if self.audio:
                self.audio.paused = False

    # ── VOICE OUTPUT ─────────────────────────────────────────────────

    async def _speak(self, text: str) -> None:
        """Speak text aloud via TTS. Pauses mic to avoid feedback loop."""
        if not self.voice.is_available:
            return
        if self.audio:
            self.audio.paused = True
        try:
            await self.voice.speak(text, wait=True)
        finally:
            if self.audio:
                self.audio.paused = False

    # ── PUSH-TO-TALK (Shift+Z) ──────────────────────────────────────

    def _start_push_to_talk(self) -> None:
        """Start the global hotkey listener for push-to-talk (Shift+Z).
        Runs pynput's keyboard listener on its own thread."""
        try:
            from pynput import keyboard
        except ImportError:
            print("   [PTT] pynput not installed — push-to-talk disabled")
            return

        self._ptt_keys_held: set = set()
        self._ptt_recording = False
        self._ptt_audio_chunks: list = []

        def on_press(key):
            from pynput import keyboard as kb
            try:
                if key == kb.Key.shift or key == kb.Key.shift_l or key == kb.Key.shift_r:
                    self._ptt_keys_held.add("shift")
                elif hasattr(key, 'char') and key.char and key.char.lower() == 'z':
                    self._ptt_keys_held.add("z")
            except AttributeError:
                pass

            if "shift" in self._ptt_keys_held and "z" in self._ptt_keys_held:
                if not self._ptt_recording:
                    self._ptt_recording = True
                    self._ptt_audio_chunks = []
                    self.audio._play_listening_tone()
                    print(f"\n   🎤 PUSH-TO-TALK — Recording! (hold Shift+Z, release when done)")

        def on_release(key):
            from pynput import keyboard as kb
            try:
                if key == kb.Key.shift or key == kb.Key.shift_l or key == kb.Key.shift_r:
                    self._ptt_keys_held.discard("shift")
                elif hasattr(key, 'char') and key.char and key.char.lower() == 'z':
                    self._ptt_keys_held.discard("z")
            except AttributeError:
                pass

            if self._ptt_recording and ("shift" not in self._ptt_keys_held or "z" not in self._ptt_keys_held):
                self._ptt_recording = False
                print(f"   🎤 Released — processing...")

        from pynput import keyboard as kb
        self._ptt_listener = kb.Listener(on_press=on_press, on_release=on_release)
        self._ptt_listener.daemon = True
        self._ptt_listener.start()
        print(f"   Push-to-talk: Hold Shift+Z to talk directly!")

    async def _ptt_recording_loop(self) -> None:
        """Records audio while push-to-talk keys are held, then transcribes.
        Uses one continuous sd.rec() call, stopped with sd.stop() on release,
        to avoid the choppy-chunks problem with low-gain mics."""
        import sounddevice as sd

        while self.heartbeat.is_running:
            if not self._ptt_recording:
                await asyncio.sleep(0.05)
                continue

            if self.audio and self.audio.is_available:
                self.audio.paused = True

            max_seconds = 30
            total_samples = int(max_seconds * self.audio.sample_rate)
            audio_buf = sd.rec(total_samples, samplerate=self.audio.sample_rate,
                               channels=1, dtype="float32")

            while self._ptt_recording:
                await asyncio.sleep(0.05)

            sd.stop()

            if self.audio:
                self.audio.paused = False

            audio_data = audio_buf.flatten()
            last_nonzero = len(audio_data)
            while last_nonzero > 0 and abs(audio_data[last_nonzero - 1]) < 1e-10:
                last_nonzero -= 1
            audio_data = audio_data[:max(last_nonzero, 1)]

            duration = len(audio_data) / self.audio.sample_rate
            if duration < 0.3:
                continue

            print(f"   🎤 Recorded {duration:.1f}s — transcribing...")

            self.audio._play_listening_tone()
            transcript = await self.audio.transcribe(audio_data)

            if not transcript or len(transcript.strip().strip(".")) < 4:
                print(f"   🎤 Couldn't make out what you said.")
                continue

            print(f"   🎤 >> \"{transcript}\"")

            message = transcript.strip()
            for phrase in self.detector.wake_phrases:
                idx = message.lower().find(phrase)
                if idx != -1:
                    message = message[idx + len(phrase):].strip().lstrip(",.!? ")
                    break

            await self._handle_direct_address(message or transcript)

    # Removed _ptt_record_chunk — no longer needed with continuous recording

    # ── BACKGROUND LISTENER (Overheard Context) ──────────────────

    async def _listening_loop(self) -> None:
        """
        Background listener for passive context (overheard speech).
        Also still detects wake word "Hey Serendipity" as a fallback.
        Primary direct interaction is via push-to-talk (Shift+Z).
        """
        self._listening = True
        listen_count = 0

        print(f"\n   [Listener] Background listener ACTIVE")

        while self.heartbeat.is_running and self._listening:
            if self._thinking or (self.audio and self.audio.paused):
                await asyncio.sleep(0.3)
                continue

            if self._ptt_recording:
                await asyncio.sleep(0.2)
                continue

            if not self.audio or not self.audio.is_available:
                await asyncio.sleep(0.5)
                continue

            try:
                listen_count += 1
                t_start = time.time()

                loop = asyncio.get_event_loop()
                audio_data = await loop.run_in_executor(
                    None, self._listen_capture
                )

                if audio_data is None:
                    if listen_count % 20 == 0:
                        print(f"   [Listener] ... still listening (cycle #{listen_count})")
                    continue

                t_captured = time.time()
                transcript = await self.audio.transcribe(audio_data)
                t_transcribed = time.time()

                if not transcript:
                    continue

                cleaned = transcript.strip().strip(".")
                if len(cleaned) < 4 or cleaned in ("you", "You", "the", "a", "I"):
                    continue

                capture_ms = int((t_captured - t_start) * 1000)
                transcribe_ms = int((t_transcribed - t_captured) * 1000)
                print(f"   [Listener] >> \"{transcript}\" "
                      f"({capture_ms}ms capture, {transcribe_ms}ms transcribe)")

                was_in_conversation = self.detector.in_conversation
                result = self.detector.detect(transcript, self.current_context)
                reason = ""
                if result.wake_word_found:
                    reason = " (wake word found!)"
                elif result.mode == "DIRECT":
                    reason = " (conversation still active)"
                print(f"   [Listener] Classified as: {result.mode}{reason}")

                if result.mode == "DIRECT":
                    await self._handle_direct_address(result.message or transcript)
                elif result.mode == "OVERHEARD":
                    if was_in_conversation:
                        print(f"   [Listener] Conversation ended -- back to listening for 'Hey Serendipity'")
                    if transcript.strip():
                        self._overheard_buffer.append(transcript.strip())
                        if len(self._overheard_buffer) > 5:
                            self._overheard_buffer = self._overheard_buffer[-5:]

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"   [Listener] Error: {e}")
                await asyncio.sleep(0.5)

    def _listen_capture(self):
        """
        Record a fixed-duration clip and check if it contains any speech.
        Returns the audio array if speech detected, None otherwise.
        Runs in an executor thread.
        """
        if not self.audio or not self.audio.is_available:
            return None
        if self.audio.paused:
            return None

        audio = self.audio.capture_audio(quiet=True)
        if audio is None:
            return None

        if not self.audio.has_speech(audio):
            return None

        return audio

    PICTURE_PHRASES = [
        "take a picture", "take a photo", "take a pic", "snap a picture",
        "snap a photo", "snap a pic", "look at this", "look at that",
        "what do you see", "what are you looking at", "check this out",
        "look over here", "see this", "see what", "take a look",
        "can you see", "do you see", "what's in front of you",
    ]

    SEARCH_PHRASES = [
        "search for", "search about", "look up", "google",
        "internet search", "web search", "find out about",
        "do a search", "do an internet search", "do a web search",
        "search the web", "search the internet", "search online",
        "what does the internet say", "find me info",
        "research", "look into",
        "do an interview", "doing an interview", "do an inner view",
        "do an inner search", "doing an inner search",
        "do an intern search", "do an internet surge",
        "tell me about", "what do you know about",
    ]

    async def _handle_direct_address(self, message: str) -> None:
        """The user said 'Hey Serendipity' — respond immediately."""
        print(f"\n   +{'=' * 48}+")
        print(f"   |  DIRECT ADDRESS DETECTED                     |")
        print(f"   +{'=' * 48}+")
        print(f"   User said: \"{message[:70]}\"")

        if not message.strip():
            print(f"\n   [Heard the wake word but no message -- listening for follow-up...]")
            follow_up = await self._listen_for_follow_up()
            if follow_up:
                message = follow_up
            else:
                print("   [No follow-up detected]")
                return

        image_description = ""
        search_facts = ""
        search_sources: list[str] = []

        if self._is_picture_request(message):
            image_description = await self._take_picture_on_demand(message)

        if self._is_search_request(message):
            search_facts, search_sources = await self._search_on_demand(message)

        print(f"   Thinking...")
        reply = await self.responder.respond(
            message, self.current_context,
            image_description=image_description,
            search_context=search_facts,
        )

        print(f"\n{'═' * 70}")
        print(f"💬 SERENDIPITY RESPONDS:\n")
        print(f"   \"{reply}\"")
        print(f"\n{'═' * 70}")
        if search_sources:
            print(f"🔍 Sources:")
            for url in search_sources:
                print(f"   📎 {url}")
            print()
        await self._speak(reply)

    def _is_picture_request(self, message: str) -> bool:
        """Check if the user is asking the engine to look at something."""
        lower = message.lower()
        return any(phrase in lower for phrase in self.PICTURE_PHRASES)

    def _is_search_request(self, message: str) -> bool:
        """Check if the user is asking for a web search."""
        lower = message.lower()
        return any(phrase in lower for phrase in self.SEARCH_PHRASES)

    async def _take_picture_on_demand(self, message: str) -> str:
        """Capture a photo and describe it. Returns the image description."""
        if not self.vision or not self.vision.is_available:
            print("   [No camera available — responding without vision]")
            return ""

        print("   [CAMERA ON]  Snapping picture on demand...")
        image_bytes = self.vision.capture_frame()
        print("   [CAMERA OFF] Got it!")

        if image_bytes is None:
            print("   [Camera capture failed]")
            return ""

        print("   [Describing what I see...]")
        description = await self.vision._describe_image(image_bytes, self.llm)
        if description:
            print(f"   [Vision] I see: {description[:80]}")
        return description

    async def _search_on_demand(self, message: str) -> tuple[str, list[str]]:
        """Run a web search based on what the user asked. Returns (facts_text, source_urls)."""
        print(f"   🔍 Searching the web for you...")
        try:
            result = await self.searcher.search_direct(message)
            if result.facts:
                facts_text = "\n".join(f"- {fact}" for fact in result.facts)
                print(f"   ✅ Found {len(result.facts)} facts from {len(result.source_urls)} sources")
                return facts_text, result.source_urls
            else:
                print(f"   ℹ️  Search completed but no specific facts extracted")
                return "", result.source_urls
        except Exception as e:
            print(f"   ⚠️  Search failed: {e}")
            return "", []

    async def _listen_for_follow_up(self) -> str:
        """After hearing just the wake word, listen for the actual question."""
        if not self.audio or not self.audio.is_available:
            return ""
        print(f"   [MIC ON]  Listening for your question...")
        transcript = await self.audio.quick_capture_and_transcribe()
        print(f"   [MIC OFF] Got it.")
        return transcript

    # ── USER INPUT LOOP ──────────────────────────────────────────────

    async def _input_loop(self) -> None:
        """Listen for user input while the heartbeat runs in the background."""
        loop = asyncio.get_event_loop()

        while self.heartbeat.is_running:
            try:
                user_input = await loop.run_in_executor(None, input, "")
                user_input = user_input.strip()
            except (EOFError, KeyboardInterrupt):
                self.heartbeat.stop()
                return

            if not user_input:
                continue

            if user_input.replace("z", "").replace("Z", "") == "":
                continue

            cmd = user_input.lower()

            if cmd in ("quit", "exit", "q"):
                self.heartbeat.stop()
                return

            elif cmd in ("not now", "notnow", "shh", "quiet", "hush"):
                self.heartbeat.backoff(2)
                print("   🤫 Okay, I'll be quiet for a bit!")

            elif cmd == "status":
                remaining = self.heartbeat.time_until_next
                mins = remaining // 60
                secs = remaining % 60
                thinking = "🧠 Thinking..." if self._thinking else "😌 Idle"
                listening = "🎙️ Active" if self._listening else "Off"
                voice_st = f"🔊 {self.voice.cfg.voice}" if self.voice.is_available else "🔇 Off"
                print(f"\n   📊 Status:")
                print(f"      State: {thinking}")
                print(f"      Listener: {listening}")
                print(f"      Voice: {voice_st}")
                print(f"      Context: \"{self.current_context}\"")
                print(f"      Heartbeats fired: {self.heartbeat.beat_count}")
                print(f"      Next heartbeat: ~{mins}m {secs}s")
                print(f"      Tempo: {self.heartbeat.tempo_info}")
                print(f"      Past topics: {len(self.past_topics)}")
                print(f"      Overheard buffer: {len(self._overheard_buffer)} items")
                print(f"      Conversation turns: {len(self.responder.history)}")
                print(f"      Memory: {self.memory.chain_count} chains stored")
                print(f"      Incubating: {self.incubation.queue_size} ideas")
                print(f"      Profile: {'✅ built' if self.profile.has_profile else '⏳ needs more ratings'}")
                if self.cocreator.is_active:
                    print(f"      Brainstorm: 🤝 active ({len(self.cocreator.session.turns)} turns)")
                if self.bridge.last_personality_name:
                    print(f"      Last personality: {self.bridge._last_personality.emoji} {self.bridge.last_personality_name}")

            elif cmd == "mute":
                self.voice.cfg.enabled = False
                self.voice._available = False
                print("   🔇 Voice muted — text only mode")

            elif cmd == "unmute":
                self.voice.cfg.enabled = True
                self.voice._available = True
                print(f"   🔊 Voice unmuted — speaking as {self.voice.cfg.voice}")

            elif cmd.startswith("voice "):
                new_voice = cmd.split(" ", 1)[1].strip().lower()
                from src.output.voice import VOICE_DESCRIPTIONS
                if new_voice == "list":
                    VoiceOutput.list_voices()
                elif new_voice in VOICE_DESCRIPTIONS:
                    self.voice.cfg.voice = new_voice
                    print(f"   🔊 Voice changed to: {new_voice} ({VOICE_DESCRIPTIONS[new_voice]})")
                    await self._speak(f"Hey! This is my {new_voice} voice. How do I sound?")
                else:
                    print(f"   ❓ Unknown voice '{new_voice}'. Available: {', '.join(VOICE_DESCRIPTIONS.keys())}")

            elif cmd == "fire":
                if self._thinking:
                    print("   ⏳ Already thinking — hang on!")
                else:
                    print("   ⚡ Forcing heartbeat NOW! (creative mode)")
                    self._force_creative = True
                    self.heartbeat._remaining_seconds = 0

            elif cmd in ("reveal", "show chain", "causation", "why"):
                if self._last_interjection:
                    self._print_causal_chain(self._last_interjection)
                else:
                    print("   ❌ No interjection to reveal yet")

            elif cmd in ("transparency", "transparency on", "debug on"):
                self._transparency = True
                print("   🔮 Transparency ON — causal chains will be shown after every interjection")
                print("   (The illusion of spontaneity dissolves. You see the machinery now.)")

            elif cmd in ("transparency off", "debug off", "magic"):
                self._transparency = False
                print("   ✨ Transparency OFF — back to the illusion of spontaneous thought")

            elif cmd == "stats":
                await self._show_serendipity_stats()

            elif cmd in ("mode deep_thought", "mode deep thought", "deep thought", "lsd mode", "deep thought on"):
                if not self._deep_thought_active:
                    self._deep_thought_active = True
                    if not self.multi_seed:
                        self.multi_seed = MultiSeedGenerator(
                            llm=self.llm,
                            tree_config=self.cfg.association_tree,
                            dt_config=self.cfg.deep_thought,
                            embedder=self.embedder,
                            memory=self.memory,
                        )
                    print(f"   🔮 DEEP THOUGHT MODE ACTIVATED")
                    print(f"   Parallel seeds: {self.cfg.deep_thought.parallel_seeds} | Max depth: {self.cfg.deep_thought.max_depth} | Min crossings: {self.cfg.deep_thought.min_domain_crossings}")
                    print(f"   The engine will now generate {self.cfg.deep_thought.parallel_seeds} parallel chains per heartbeat")
                    print(f"   and search for butterfly-effect connections between them.")
                else:
                    print(f"   🔮 Deep Thought mode is already active")

            elif cmd in ("mode normal", "normal mode", "deep thought off", "lsd off"):
                self._deep_thought_active = False
                print(f"   🌳 Normal mode restored — friendly creative companion")

            elif cmd == "mode":
                mode = "🔮 Deep Thought" if self._deep_thought_active else "🌳 Normal"
                print(f"   Current mode: {mode}")
                print(f"   Switch: 'deep thought' or 'normal mode'")

            elif cmd in ("👍", "good", "like", "thumbs up", "nice"):
                if self.memory.rate_last_interjection(5):
                    print("   ⭐ Rated last interjection 5/5 — I'll remember you liked that!")
                    self.profile.on_rating()
                    if await self.profile.rebuild_if_needed():
                        self.bridge.persona = self.profile.persona_injection
                else:
                    print("   ❌ Nothing to rate yet")

            elif cmd in ("👎", "bad", "dislike", "thumbs down", "meh"):
                if self.memory.rate_last_interjection(1):
                    print("   📝 Rated last interjection 1/5 — noted, I'll adjust")
                    self.profile.on_rating()
                    if await self.profile.rebuild_if_needed():
                        self.bridge.persona = self.profile.persona_injection
                else:
                    print("   ❌ Nothing to rate yet")

            elif cmd.startswith("rate "):
                try:
                    rating = int(cmd.split(" ", 1)[1].strip())
                    if 1 <= rating <= 5:
                        if self.memory.rate_last_interjection(rating):
                            stars = "⭐" * rating
                            print(f"   {stars} Rated {rating}/5 — stored in memory")
                            self.profile.on_rating()
                            if await self.profile.rebuild_if_needed():
                                self.bridge.persona = self.profile.persona_injection
                        else:
                            print("   ❌ Nothing to rate yet")
                    else:
                        print("   ❌ Use 'rate 1' through 'rate 5'")
                except ValueError:
                    print("   ❌ Use 'rate 1' through 'rate 5'")

            elif self.cocreator.is_active:
                if CoCreator.is_exit(user_input):
                    turns = len(self.cocreator.session.turns)
                    self.cocreator.end_session()
                    print(f"   🤝 Brainstorm ended ({turns} exchanges). Back to normal mode!")
                else:
                    await self._handle_cocreation(user_input)

            elif CoCreator.is_trigger(user_input) and self._last_interjection:
                await self._start_cocreation(user_input)

            else:
                typed_result = self.detector.detect(user_input, self.current_context)
                if typed_result.mode == "DIRECT":
                    await self._handle_direct_address(typed_result.message or user_input)
                else:
                    self.current_context = user_input
                    self.incubation.set_context(user_input)
                    print(f"   ✅ Context updated: \"{self.current_context}\"")

    # ── CO-CREATION MODE ─────────────────────────────────────────────

    async def _start_cocreation(self, trigger_text: str) -> None:
        """Enter co-creation mode based on the last interjection."""
        interjection = self._last_interjection
        chain = interjection.chain if interjection else None

        self.cocreator.start_session(
            interjection_text=interjection.interjection_text if interjection else "",
            seed_topic=chain.seed_topic if chain else "",
            endpoint_topic=chain.endpoint_topic if chain else "",
            chain_summary=chain.summary() if chain else "",
        )

        print(f"\n   +{'=' * 48}+")
        print(f"   |  🤝 CO-CREATION MODE ACTIVATED               |")
        print(f"   |  Brainstorming together! Say 'done' to exit.  |")
        print(f"   +{'=' * 48}+")

        if self.audio:
            self.audio.paused = True

        reply = await self.cocreator.respond(trigger_text, self.current_context)

        print(f"\n{'═' * 70}")
        print(f"🤝 BRAINSTORM:\n")
        print(f"   \"{reply}\"")
        print(f"\n{'═' * 70}")
        self.responder.add_engine_interjection(reply)
        await self._speak(reply)

        if self.audio:
            self.audio.paused = False

    async def _handle_cocreation(self, user_input: str) -> None:
        """Continue an active co-creation session."""
        if self.audio:
            self.audio.paused = True

        print(f"   🤝 You said: \"{user_input[:70]}\"")
        reply = await self.cocreator.respond(user_input, self.current_context)

        print(f"\n{'═' * 70}")
        print(f"🤝 BRAINSTORM:\n")
        print(f"   \"{reply}\"")
        print(f"\n{'═' * 70}")
        self.responder.add_engine_interjection(reply)
        await self._speak(reply)

        if self.audio:
            self.audio.paused = False

    # ── SINGLE-FIRE & INTERACTIVE MODES (unchanged) ──────────────────

    def _print_citations(self, interjection: Interjection) -> None:
        """Print source URLs and facts from an interjection's web search."""
        if interjection.search_facts:
            print(f"🔍 Grounded in {len(interjection.search_facts)} web facts"
                  f"{f' from {len(interjection.search_sources)} sources' if interjection.search_sources else ''}")
        if interjection.search_sources:
            for url in interjection.search_sources[:5]:
                print(f"   📎 {url}")

    def _print_causal_chain(self, interjection: Interjection) -> None:
        """Transparency mode: reveal the full causal chain that produced this interjection.

        This proves the core thesis — every 'creative' output has a deterministic
        causal chain behind it. Hidden, it feels like spontaneous thought.
        Revealed, you see the machinery. Same output, different experience.
        """
        print(f"\n{'─' * 70}")
        print(f"🔮 CAUSAL CHAIN REVEALED (transparency mode)")
        print(f"{'─' * 70}")

        chain = interjection.chain
        scoring = interjection.scoring
        ctx = interjection.context

        print(f"   ┌─ TRIGGER")
        print(f"   │  Heartbeat #{interjection.heartbeat_id} fired")
        if ctx and ctx.dominant_channel:
            print(f"   │  Dominant input: {ctx.dominant_channel} (novelty: {ctx.overall_novelty:.2f})")
        print(f"   │")

        print(f"   ├─ SEED TOPIC")
        print(f"   │  \"{ctx.seed_topic if ctx else 'unknown'}\"")
        print(f"   │")

        if chain and chain.nodes:
            print(f"   ├─ ASSOCIATION CHAIN ({len(chain.nodes)} hops, {chain.domain_crossings} domain crossings)")
            for i, node in enumerate(chain.nodes):
                connector = "│  " if i < len(chain.nodes) - 1 else "│  "
                arrow = "→" if i > 0 else "●"
                domain_tag = f"  [{node.domain}]" if i == 0 or node.domain != chain.nodes[i-1].domain else ""
                print(f"   │  {arrow} {node.topic}{domain_tag}")
                if i > 0 and node.connection_reason:
                    print(f"   │    ↳ why: {node.connection_reason[:70]}")
            if chain.total_semantic_distance > 0:
                print(f"   │  Embedding distance (seed→endpoint): {chain.total_semantic_distance:.3f}")
            print(f"   │")

        if scoring:
            print(f"   ├─ SCORING BREAKDOWN")
            print(f"   │  semantic_distance : {scoring.semantic_distance:.3f} (×0.30 = {scoring.semantic_distance * 0.30:.3f})")
            print(f"   │  domain_crossings  : {scoring.domain_crossings:.3f} (×0.25 = {scoring.domain_crossings * 0.25:.3f})")
            print(f"   │  surprise          : {scoring.surprise:.3f} (×0.20 = {scoring.surprise * 0.20:.3f})")
            print(f"   │  bridgeability     : {scoring.bridgeability:.3f} (×0.15 = {scoring.bridgeability * 0.15:.3f})")
            print(f"   │  novelty           : {scoring.novelty:.3f} (×0.10 = {scoring.novelty * 0.10:.3f})")
            print(f"   │  TOTAL             : {scoring.total:.3f}")
            print(f"   │")

        personality = self.bridge._last_personality
        if personality:
            print(f"   ├─ PERSONALITY SELECTED")
            print(f"   │  {personality.emoji} {personality.name}: {personality.description}")
            print(f"   │")

        print(f"   └─ OUTPUT")
        print(f"      \"{interjection.interjection_text[:100]}{'...' if len(interjection.interjection_text) > 100 else ''}\"")
        print(f"\n   💡 The causation is complex. The output feels spontaneous.")
        print(f"      That's the whole point.")
        print(f"{'─' * 70}")

    async def _show_serendipity_stats(self) -> None:
        """Display the serendipity self-evaluation report."""
        if not self.memory.is_available:
            print("   ❌ Memory not available — stats require ChromaDB")
            return

        report = self.analytics.generate_report()
        print(self.analytics.format_report(report))

        if report.total_interjections >= 5:
            session_report = self.analytics.generate_report(hours=1)
            if session_report.total_interjections >= 2:
                print(f"\n   📊 THIS SESSION ({session_report.period_label}):")
                print(f"   │  {session_report.total_interjections} interjections, "
                      f"{session_report.aha_count} AHA! moments ({session_report.aha_rate:.1f}%)")
                print(f"   │  Avg score: {session_report.avg_score:.3f}")

    async def run_single(self, seed_topic: str) -> None:
        """Fire a single heartbeat cycle — for testing and demos."""
        self.embedder.initialize()
        print("=" * 70)
        print("🧠 COMPUTATIONAL SERENDIPITY — Proof of Concept")
        print("=" * 70)

        ctx = await self.heartbeat.fire_once(seed_topic)
        t0 = time.time()

        print(f"\n🌳 Generating association tree from: \"{seed_topic}\"")
        print(f"   Config: branching={self.cfg.association_tree.branching_factor}, "
              f"depth={self.cfg.association_tree.min_depth}-{self.cfg.association_tree.max_depth}, "
              f"pruning=keep top {self.cfg.association_tree.keep_per_level}/level")

        chains = await self.tree_gen.generate_tree(seed_topic)

        if not chains:
            print("\n❌ No association chains were generated. Try a different seed topic.")
            return

        print(f"\n📊 Scoring top candidates from {len(chains)} chains...")
        ranked = await self.scorer.rank_chains(chains, ctx, self.past_topics)

        print(f"\n{'─' * 70}")
        print("📋 TOP CANDIDATES:\n")
        for i, (chain, score) in enumerate(ranked, 1):
            status = "✅ FIRE" if score.total >= self.cfg.scoring.fire_threshold else (
                "🧪 INCUBATE" if score.total >= self.cfg.scoring.incubation_threshold else "❌ DISCARD"
            )
            print(f"  #{i} [{status}] Score: {score.total:.3f}")
            print(f"     Chain: {chain.summary()}")
            embed_tag = f" [cosine: {chain.total_semantic_distance:.3f}]" if chain.total_semantic_distance > 0 else ""
            print(f"     Breakdown: dist={score.semantic_distance:.2f} cross={score.domain_crossings:.2f} "
                  f"surprise={score.surprise:.2f} bridge={score.bridgeability:.2f} novel={score.novelty:.2f}"
                  f"{embed_tag}")
            print()

        best_chain, best_score = ranked[0]

        if best_score.total >= self.cfg.scoring.fire_threshold:
            print(f"{'─' * 70}")
            search_result = await self._search_for_facts(best_chain, ctx)

            print(f"\n🌉 Building interjection from best chain + search results...\n")
            interjection = await self.bridge.build_interjection(
                best_chain,
                best_score,
                ctx,
                search_facts=search_result.facts if search_result else None,
                search_sources=search_result.source_urls if search_result else None,
            )
            self.past_topics.append(best_chain.endpoint_topic)
            self.scorer.record_interjection(best_chain)

            elapsed = time.time() - t0
            print(f"{'═' * 70}")
            print(f"💬 SERENDIPITY SAYS:\n")
            print(f"   \"{interjection.interjection_text}\"")
            print(f"\n{'═' * 70}")
            print(f"\n⏱️  Total pipeline time: {elapsed:.1f}s")
            print(f"📍 Internal chain: {best_chain.summary()}")
            print(f"📊 Interest score: {best_score.total:.3f}")
            self._print_citations(interjection)
        else:
            elapsed = time.time() - t0
            print(f"{'─' * 70}")
            print(f"🤫 Nothing interesting enough to share this cycle.")
            print(f"   Best score: {best_score.total:.3f} (threshold: {self.cfg.scoring.fire_threshold})")
            print(f"   Best chain: {best_chain.summary()}")
            print(f"\n⏱️  Total pipeline time: {elapsed:.1f}s")

    async def _search_for_facts(self, chain, ctx):
        """Run web search on the winning chain's endpoint."""
        print(f"🔍 Searching the web for facts about: \"{chain.endpoint_topic}\"")
        try:
            search_result = await self.searcher.search_for_chain(
                endpoint_topic=chain.endpoint_topic,
                chain_summary=chain.summary(),
                context=ctx.seed_topic,
            )
            if search_result.facts:
                print(f"   ✅ Found {len(search_result.facts)} facts:")
                for fact in search_result.facts:
                    print(f"      • {fact[:100]}{'...' if len(fact) > 100 else ''}")
            else:
                print(f"   ℹ️  No specific facts extracted from search results")
            return search_result
        except Exception as e:
            print(f"   ⚠️  Search failed: {e}")
            return None

    async def run_interactive(self) -> None:
        """Interactive mode — enter seed topics and watch the engine think."""
        self.embedder.initialize()
        print("=" * 70)
        print("🧠 COMPUTATIONAL SERENDIPITY — Interactive Mode")
        print("=" * 70)
        print(f"\nProvider: {self.cfg.llm.provider} | Model: {self.cfg.llm.model}")
        search_status = "✅ Tavily" if self.searcher.is_available else "⚠️  No API key (using LLM fallback)"
        print(f"Search: {search_status}")
        print("Type a seed topic and press Enter to fire a heartbeat cycle.")
        print("Type 'quit' to exit.\n")

        while True:
            try:
                topic = input("🎯 Seed topic: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n👋 Computational Serendipity shutting down. Stay creative!")
                break

            if not topic:
                continue
            if topic.lower() in ("quit", "exit", "q"):
                print("\n👋 Computational Serendipity shutting down. Stay creative!")
                break

            await self.run_single(topic)
            print()


def print_devices():
    """List all available cameras and microphones."""
    print("=" * 70)
    print("   AVAILABLE DEVICES")
    print("=" * 70)

    print("\n   CAMERAS (Vision):")
    print("   " + "-" * 50)
    cameras = VisionChannel.list_devices()
    if cameras:
        for cam in cameras:
            status = "OK" if cam["working"] else "FOUND (no frame)"
            print(f"   [{cam['index']}] {cam['name']}  -- {status}")
    else:
        print("   No cameras detected (is opencv-python installed?)")

    print("\n   MICROPHONES (Audio):")
    print("   " + "-" * 50)
    mics = AudioChannel.list_devices()
    if mics:
        for mic in mics:
            default_tag = " <-- DEFAULT" if mic["is_default"] else ""
            print(f"   [{mic['index']}] {mic['name']}")
            print(f"        channels={mic['channels']}, rate={mic['sample_rate']}Hz{default_tag}")
    else:
        print("   No microphones detected (is sounddevice installed?)")

    print("\n   " + "-" * 50)
    print("   To select devices, add to your config.yaml:\n")
    print("   input_pipeline:")
    print("     vision:")
    print("       device_index: 0    # camera number from list above")
    print("     audio:")
    print("       device_index: 0    # microphone number from list above")
    print()


async def main():
    args = sys.argv[1:]

    if "--devices" in args:
        print_devices()
        return

    debug_audio = "--debug-audio" in args
    if debug_audio:
        args.remove("--debug-audio")

    config = load_config()

    deep_thought_cli = False
    if "--mode" in args:
        mode_idx = args.index("--mode")
        if mode_idx + 1 < len(args):
            mode_value = args[mode_idx + 1].lower().replace(" ", "_")
            if mode_value in ("deep_thought", "deep-thought", "deepthought", "lsd"):
                config.creativity_mode = "deep_thought"
                config.deep_thought.enabled = True
                deep_thought_cli = True
            args.pop(mode_idx + 1)
            args.pop(mode_idx)

    engine = ComputationalSerendipity(config, debug_audio=debug_audio)

    if deep_thought_cli:
        print(f"\n   🔮 DEEP THOUGHT MODE via CLI flag")

    if "--live" in args:
        args.remove("--live")
        initial_context = " ".join(args) if args else ""
        await engine.run_live(initial_context)
    elif args:
        seed = " ".join(args)
        await engine.run_single(seed)
    else:
        await engine.run_interactive()


if __name__ == "__main__":
    asyncio.run(main())
