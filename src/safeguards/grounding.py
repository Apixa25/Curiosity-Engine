"""
Epistemological Safeguards — Distinguishing Insight from Hallucination.

Deep Thought mode generates chains so long and cross-domain that no human
can trace them. This makes verification critical. Without safeguards, the
engine could produce convincing-sounding hypotheses built entirely on LLM
confabulation. With them, we get calibrated confidence levels.

The safeguard pipeline:
1. GROUNDING — Each hop in a chain gets a quick web check via Tavily
2. TAGGING — Nodes are labeled: "grounded", "inferred", or "speculative"
3. CONFIDENCE — The ratio of grounded-to-speculative determines confidence
4. MECHANISM VERIFICATION — Collision points must name specific mechanisms
5. PREDICTION EXTRACTION — Hypotheses must include testable predictions

Confidence Levels:
  HIGH   — 70%+ hops grounded, mechanism is published and specific
  MEDIUM — 40-70% grounded, mechanism plausible but not directly studied
  LOW    — <40% grounded, mechanism speculative, but collision is real
  ORACLE — Chain too long to verify, but collision strength very high
           (the "42" moment: answer feels right, proof is too complex)
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field

from src.config.llm_adapter import LLMAdapter
from src.models import AssociationChain, AssociationNode, CollisionResult, CollisionScore


GROUNDING_QUERY_TEMPLATE = """Given this causal connection in a chain of reasoning:

Previous concept: "{from_topic}" ({from_domain})
Connection: "{connection_reason}"
Next concept: "{to_topic}" ({to_domain})

Construct a SHORT, specific web search query (5-10 words) that could verify whether
this causal connection is real. Focus on the MECHANISM — how does A actually lead to B?

Return ONLY the search query, nothing else."""

GROUNDING_VERIFY_TEMPLATE = """I need to verify whether this causal connection is real:

"{from_topic}" → "{to_topic}"
Claimed connection: "{connection_reason}"

Here is what a web search found:
{search_results}

Based on the search results, classify this connection:
- GROUNDED: The search results directly support this connection with specific evidence
- INFERRED: The search results support related mechanisms but don't directly confirm this exact link
- SPECULATIVE: No meaningful support found; this is plausible but unverified

Return ONLY one word: GROUNDED, INFERRED, or SPECULATIVE"""

MECHANISM_VERIFY_TEMPLATE = """Two independent chains of thought collided at a hidden intersection:

Chain A topic near collision: "{node_a_topic}" ({node_a_domain})
Chain B topic near collision: "{node_b_topic}" ({node_b_domain})
Claimed shared concept: "{collision_concept}"

What is the specific physical/chemical/biological/social MECHANISM by which these two
concepts are causally connected? Not just "they're related" — name the actual causal pathway.

If you can name a specific, published mechanism: return it in 1-2 sentences.
If the mechanism is plausible but not published: say "PLAUSIBLE: " then describe it.
If there is no real mechanism: say "NONE" and explain why this is a spurious connection.

Be honest. Most collisions are interesting correlations, not verified causal mechanisms."""

PREDICTION_EXTRACT_TEMPLATE = """Given this hypothesis generated from a collision between two causal chains:

HYPOTHESIS:
{hypothesis}

COLLISION CONCEPT: "{collision_concept}"
CONFIDENCE: {confidence}

Extract a TESTABLE PREDICTION from this hypothesis. Format:
"If [this hidden link is real], then [observable prediction when you do specific action]"

The prediction should be:
- SPECIFIC (not "things would be different")
- ACTIONABLE (someone could actually test this)
- FALSIFIABLE (there must be a way to disprove it)

If the hypothesis is too vague to extract a testable prediction, return:
"UNFALSIFIABLE: " followed by why.

Return ONLY the prediction statement, nothing else."""


@dataclass
class GroundingReport:
    """Summary of epistemological verification for a chain or collision."""
    total_hops: int = 0
    grounded_hops: int = 0
    inferred_hops: int = 0
    speculative_hops: int = 0
    unverified_hops: int = 0
    grounding_ratio: float = 0.0
    confidence_level: str = "low"
    mechanism_verified: bool = False
    mechanism_description: str = ""
    testable_prediction: str = ""
    is_falsifiable: bool = True
    verification_sources: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Grounding: {self.grounded_hops}/{self.total_hops} hops verified "
            f"({self.grounding_ratio:.0%})\n"
            f"Confidence: {self.confidence_level.upper()}\n"
            f"Mechanism: {'✓ ' + self.mechanism_description[:80] if self.mechanism_verified else '✗ Unverified'}\n"
            f"Prediction: {self.testable_prediction[:100] if self.testable_prediction else 'None extracted'}"
        )


class EpistemologicalEngine:
    """Verifies chains and collisions against external evidence.

    Uses the existing Tavily web search integration to ground individual
    hops, then aggregates into a confidence level. Also extracts and
    validates testable predictions from hypotheses.
    """

    def __init__(self, llm: LLMAdapter, api_key_env: str = "TAVILY_API_KEY"):
        self.llm = llm
        self.api_key = os.environ.get(api_key_env, "")
        self._client = None

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    async def _get_tavily(self):
        if self._client is None:
            from tavily import AsyncTavilyClient
            self._client = AsyncTavilyClient(api_key=self.api_key)
        return self._client

    async def ground_chain(
        self,
        chain: AssociationChain,
        max_hops_to_verify: int = 6,
        verbose: bool = False,
    ) -> GroundingReport:
        """Verify intermediate hops in a chain against web evidence.

        Doesn't verify every hop (too expensive). Samples strategically:
        - Always verify the first hop (seed → first association)
        - Always verify the last hop (penultimate → endpoint)
        - Sample from the middle based on max_hops_to_verify
        """
        if not chain.nodes or len(chain.nodes) < 2:
            return GroundingReport()

        hops_to_verify = self._select_hops_to_verify(chain.nodes, max_hops_to_verify)

        if verbose:
            print(f"   🔬 Verifying {len(hops_to_verify)} of {len(chain.nodes)-1} hops...")

        tasks = []
        for from_node, to_node in hops_to_verify:
            tasks.append(self._verify_single_hop(from_node, to_node))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        grounded = 0
        inferred = 0
        speculative = 0
        sources: list[str] = []

        for i, (result, (from_node, to_node)) in enumerate(zip(results, hops_to_verify)):
            if isinstance(result, Exception):
                to_node.grounding_status = "unverified"
                continue

            status, source = result
            to_node.grounding_status = status
            to_node.grounding_source = source

            if status == "grounded":
                grounded += 1
                if source:
                    sources.append(source)
            elif status == "inferred":
                inferred += 1
            else:
                speculative += 1

            if verbose:
                icon = {"grounded": "✓", "inferred": "~", "speculative": "✗"}.get(status, "?")
                print(f"      {icon} {from_node.topic} → {to_node.topic}: {status}")

        total = grounded + inferred + speculative
        ratio = (grounded + inferred * 0.5) / max(total, 1)
        confidence = self._compute_confidence(ratio, total, len(chain.nodes))

        return GroundingReport(
            total_hops=len(chain.nodes) - 1,
            grounded_hops=grounded,
            inferred_hops=inferred,
            speculative_hops=speculative,
            unverified_hops=(len(chain.nodes) - 1) - total,
            grounding_ratio=ratio,
            confidence_level=confidence,
            verification_sources=sources,
        )

    async def verify_collision_mechanism(
        self,
        collision: CollisionResult,
        verbose: bool = False,
    ) -> tuple[bool, str]:
        """Verify that a real causal mechanism exists at the collision point.

        Returns (is_verified, mechanism_description).
        """
        if not collision.collision_node_a or not collision.collision_node_b:
            return False, ""

        prompt = MECHANISM_VERIFY_TEMPLATE.format(
            node_a_topic=collision.collision_node_a.topic,
            node_a_domain=collision.collision_node_a.domain,
            node_b_topic=collision.collision_node_b.topic,
            node_b_domain=collision.collision_node_b.domain,
            collision_concept=collision.collision_concept,
        )

        try:
            response = await self.llm.generate(prompt, temperature=0.3)
            text = response.text.strip()

            if text.startswith("NONE"):
                if verbose:
                    print(f"      ✗ No verified mechanism at collision point")
                return False, text

            is_verified = not text.startswith("PLAUSIBLE")
            mechanism = text.replace("PLAUSIBLE: ", "").replace("PLAUSIBLE:", "")

            if verbose:
                icon = "✓" if is_verified else "~"
                print(f"      {icon} Mechanism: {mechanism[:80]}")

            return is_verified, mechanism

        except Exception:
            return False, ""

    async def extract_prediction(
        self,
        hypothesis: str,
        collision_concept: str,
        confidence: str,
    ) -> tuple[str, bool]:
        """Extract a testable prediction from a hypothesis.

        Returns (prediction_text, is_falsifiable).
        """
        prompt = PREDICTION_EXTRACT_TEMPLATE.format(
            hypothesis=hypothesis,
            collision_concept=collision_concept,
            confidence=confidence,
        )

        try:
            response = await self.llm.generate(prompt, temperature=0.4)
            text = response.text.strip()

            if text.startswith("UNFALSIFIABLE"):
                return text, False

            return text, True

        except Exception:
            return "", False

    async def full_verification(
        self,
        collision: CollisionResult,
        verbose: bool = False,
    ) -> GroundingReport:
        """Run the complete epistemological verification pipeline on a collision.

        1. Ground both chains
        2. Verify the collision mechanism
        3. Extract and validate the prediction
        4. Compute final confidence
        """
        report = GroundingReport()

        if verbose:
            print(f"\n   🔬 EPISTEMOLOGICAL VERIFICATION")

        if collision.chain_a:
            if verbose:
                print(f"   │  Verifying Chain A ({collision.seed_a_label})...")
            report_a = await self.ground_chain(collision.chain_a, verbose=verbose)
            report.grounded_hops += report_a.grounded_hops
            report.inferred_hops += report_a.inferred_hops
            report.speculative_hops += report_a.speculative_hops
            report.verification_sources.extend(report_a.verification_sources)

        if collision.chain_b:
            if verbose:
                print(f"   │  Verifying Chain B ({collision.seed_b_label})...")
            report_b = await self.ground_chain(collision.chain_b, verbose=verbose)
            report.grounded_hops += report_b.grounded_hops
            report.inferred_hops += report_b.inferred_hops
            report.speculative_hops += report_b.speculative_hops
            report.verification_sources.extend(report_b.verification_sources)

        report.total_hops = collision.total_hops

        if verbose:
            print(f"   │  Verifying collision mechanism...")
        verified, mechanism = await self.verify_collision_mechanism(collision, verbose=verbose)
        report.mechanism_verified = verified
        report.mechanism_description = mechanism

        if collision.hypothesis:
            if verbose:
                print(f"   │  Extracting testable prediction...")
            prediction, falsifiable = await self.extract_prediction(
                collision.hypothesis,
                collision.collision_concept,
                collision.confidence,
            )
            report.testable_prediction = prediction
            report.is_falsifiable = falsifiable
            if verbose:
                icon = "✓" if falsifiable else "~"
                print(f"      {icon} {prediction[:80]}")

        total_verified = report.grounded_hops + report.inferred_hops + report.speculative_hops
        if total_verified > 0:
            report.grounding_ratio = (
                report.grounded_hops + report.inferred_hops * 0.5
            ) / total_verified

        report.confidence_level = self._compute_final_confidence(
            report.grounding_ratio,
            report.mechanism_verified,
            report.is_falsifiable,
            collision.collision_similarity,
        )

        if verbose:
            print(f"   │")
            print(f"   └─ CONFIDENCE: {report.confidence_level.upper()} "
                  f"(grounding: {report.grounding_ratio:.0%}, "
                  f"mechanism: {'✓' if report.mechanism_verified else '✗'}, "
                  f"falsifiable: {'✓' if report.is_falsifiable else '✗'})")

        return report

    def _select_hops_to_verify(
        self, nodes: list[AssociationNode], max_verify: int
    ) -> list[tuple[AssociationNode, AssociationNode]]:
        """Select which hops to verify — prioritize start, end, and strategic midpoints."""
        all_hops = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]

        if len(all_hops) <= max_verify:
            return all_hops

        selected = []
        selected.append(all_hops[0])
        selected.append(all_hops[-1])

        remaining = all_hops[1:-1]
        step = max(1, len(remaining) // (max_verify - 2))
        for i in range(0, len(remaining), step):
            if len(selected) >= max_verify:
                break
            selected.append(remaining[i])

        return selected

    async def _verify_single_hop(
        self, from_node: AssociationNode, to_node: AssociationNode
    ) -> tuple[str, str]:
        """Verify a single hop: search web + classify."""
        query = await self._build_grounding_query(from_node, to_node)

        if self.is_available:
            search_text = await self._quick_search(query)
        else:
            search_text = "(web search unavailable — using LLM inference only)"

        status = await self._classify_hop(from_node, to_node, search_text)
        source = query if status == "grounded" else ""
        return status, source

    async def _build_grounding_query(
        self, from_node: AssociationNode, to_node: AssociationNode
    ) -> str:
        """Build a focused search query for verifying a single hop."""
        prompt = GROUNDING_QUERY_TEMPLATE.format(
            from_topic=from_node.topic,
            from_domain=from_node.domain,
            connection_reason=to_node.connection_reason,
            to_topic=to_node.topic,
            to_domain=to_node.domain,
        )
        try:
            resp = await self.llm.generate(prompt, temperature=0.3)
            return resp.text.strip().strip('"')
        except Exception:
            return f"{from_node.topic} {to_node.topic} mechanism"

    async def _quick_search(self, query: str, max_results: int = 3) -> str:
        """Quick Tavily search for grounding evidence."""
        try:
            client = await self._get_tavily()
            response = await client.search(
                query=query,
                search_depth="basic",
                max_results=max_results,
            )
            results = response.get("results", [])
            if not results:
                return "(no results)"

            parts = []
            for r in results[:3]:
                title = r.get("title", "")
                content = r.get("content", "")[:200]
                parts.append(f"- {title}: {content}")
            return "\n".join(parts)

        except Exception:
            return "(search failed)"

    async def _classify_hop(
        self, from_node: AssociationNode, to_node: AssociationNode, search_results: str
    ) -> str:
        """Classify a hop as grounded, inferred, or speculative."""
        prompt = GROUNDING_VERIFY_TEMPLATE.format(
            from_topic=from_node.topic,
            to_topic=to_node.topic,
            connection_reason=to_node.connection_reason,
            search_results=search_results,
        )

        try:
            resp = await self.llm.generate(prompt, temperature=0.2)
            text = resp.text.strip().upper()

            if "GROUNDED" in text:
                return "grounded"
            elif "INFERRED" in text:
                return "inferred"
            else:
                return "speculative"
        except Exception:
            return "speculative"

    def _compute_confidence(self, ratio: float, verified_count: int, chain_length: int) -> str:
        """Compute confidence level for a single chain."""
        if ratio >= 0.7:
            return "high"
        elif ratio >= 0.4:
            return "medium"
        else:
            return "low"

    def _compute_final_confidence(
        self,
        grounding_ratio: float,
        mechanism_verified: bool,
        is_falsifiable: bool,
        collision_similarity: float,
    ) -> str:
        """Compute final confidence considering all safeguard signals.

        ORACLE level is reserved for cases where:
        - Collision similarity is very high (>0.9)
        - But grounding is incomplete (chain too long/complex to fully verify)
        - The pattern is so strong it "feels right" even without full proof
        """
        if grounding_ratio >= 0.7 and mechanism_verified:
            return "high"
        elif grounding_ratio >= 0.4 and (mechanism_verified or is_falsifiable):
            return "medium"
        elif collision_similarity >= 0.90 and grounding_ratio >= 0.2:
            return "oracle"
        else:
            return "low"
