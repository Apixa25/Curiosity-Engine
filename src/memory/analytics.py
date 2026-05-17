"""
Creativity Self-Evaluation & Analytics 📊

Scores each output using Mednick/Kenett-inspired metrics and tracks
an "AHA! rate" over time, giving data-driven insights into the engine's
creative performance.

Mednick's Remote Associates Theory: creativity = connecting distant ideas.
Kenett's Network Science model: creative thinkers have more flexible semantic networks.

We measure both through our existing scoring dimensions and aggregate them
into actionable session and lifetime stats.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from collections import defaultdict

from src.memory.store import MemoryStore, StoredChain


# ── Metric thresholds (Mednick/Kenett inspired) ──────────────────────

AHA_SCORE_THRESHOLD = 0.65       # chains scoring above this are "AHA! moments"
HIGH_DISTANCE_THRESHOLD = 0.7    # semantic distance above this = genuinely remote association
CROSS_DOMAIN_THRESHOLD = 2       # 2+ domain crossings = cross-domain creative leap


@dataclass
class CreativitySnapshot:
    """Point-in-time metrics for a single creative output."""
    chain_id: str
    timestamp: float
    total_score: float
    semantic_distance: float
    domain_crossings: float
    surprise: float
    bridgeability: float
    novelty: float
    user_rating: int | None = None
    is_aha: bool = False
    is_remote_association: bool = False


@dataclass
class CreativityReport:
    """Aggregated creativity metrics over a time window."""
    total_interjections: int = 0
    aha_count: int = 0
    aha_rate: float = 0.0                  # % of interjections that are AHA! moments

    avg_score: float = 0.0
    avg_semantic_distance: float = 0.0
    avg_domain_crossings: float = 0.0
    avg_surprise: float = 0.0
    avg_bridgeability: float = 0.0
    avg_novelty: float = 0.0

    best_score: float = 0.0
    best_chain_summary: str = ""

    remote_association_count: int = 0      # chains with high semantic distance
    remote_association_rate: float = 0.0

    user_rated_count: int = 0
    avg_user_rating: float = 0.0
    user_aha_alignment: float = 0.0        # correlation between high scores and high ratings

    # Trend: is creativity improving?
    trend_direction: str = "—"             # "↑ improving", "↓ declining", "→ steady"
    trend_detail: str = ""

    # Top scoring dimensions (what's driving creativity)
    strongest_dimension: str = ""
    weakest_dimension: str = ""

    # Time range
    period_label: str = "all time"
    first_timestamp: float = 0.0
    last_timestamp: float = 0.0


class CreativityAnalytics:
    """Analyzes the engine's creative performance over time.

    Uses stored chains with their full scoring breakdowns to compute
    Mednick/Kenett-style metrics and track an AHA! rate.
    """

    def __init__(self, memory: MemoryStore):
        self.memory = memory

    def _chains_to_snapshots(self, chains: list[StoredChain]) -> list[CreativitySnapshot]:
        """Convert stored chains to creativity snapshots."""
        snapshots = []
        for c in chains:
            snap = CreativitySnapshot(
                chain_id=c.id,
                timestamp=c.timestamp,
                total_score=c.score,
                semantic_distance=c.score_semantic_distance,
                domain_crossings=c.score_domain_crossings,
                surprise=c.score_surprise,
                bridgeability=c.score_bridgeability,
                novelty=c.score_novelty,
                user_rating=c.user_rating,
                is_aha=c.score >= AHA_SCORE_THRESHOLD,
                is_remote_association=c.score_semantic_distance >= HIGH_DISTANCE_THRESHOLD,
            )
            snapshots.append(snap)
        return snapshots

    def generate_report(self, hours: float | None = None) -> CreativityReport:
        """Generate a full creativity report.

        Args:
            hours: If set, only include chains from the last N hours.
                   None = all-time report.
        """
        all_chains = self.memory.get_all_fired()
        if not all_chains:
            return CreativityReport()

        if hours is not None:
            cutoff = time.time() - (hours * 3600)
            chains = [c for c in all_chains if c.timestamp >= cutoff]
            period_label = f"last {hours:.0f}h" if hours >= 1 else f"last {hours * 60:.0f}min"
        else:
            chains = all_chains
            period_label = "all time"

        if not chains:
            return CreativityReport(period_label=period_label)

        snapshots = self._chains_to_snapshots(chains)
        n = len(snapshots)

        # Basic aggregates
        scores = [s.total_score for s in snapshots]
        aha_count = sum(1 for s in snapshots if s.is_aha)
        remote_count = sum(1 for s in snapshots if s.is_remote_association)

        avg_dims = {
            "semantic_distance": sum(s.semantic_distance for s in snapshots) / n,
            "domain_crossings": sum(s.domain_crossings for s in snapshots) / n,
            "surprise": sum(s.surprise for s in snapshots) / n,
            "bridgeability": sum(s.bridgeability for s in snapshots) / n,
            "novelty": sum(s.novelty for s in snapshots) / n,
        }

        strongest = max(avg_dims, key=avg_dims.get)
        weakest = min(avg_dims, key=avg_dims.get)

        # Best chain
        best_idx = scores.index(max(scores))
        best_chain = chains[best_idx]

        # User ratings
        rated = [s for s in snapshots if s.user_rating is not None and s.user_rating > 0]
        avg_user_rating = sum(s.user_rating for s in rated) / len(rated) if rated else 0.0

        # AHA alignment: do high-scoring chains also get high user ratings?
        aha_alignment = 0.0
        if rated:
            high_score_high_rated = sum(
                1 for s in rated if s.is_aha and s.user_rating >= 4
            )
            aha_alignment = high_score_high_rated / len(rated) if rated else 0.0

        # Trend: compare first half vs second half
        trend_direction = "→ steady"
        trend_detail = ""
        if n >= 4:
            mid = n // 2
            first_half_avg = sum(scores[:mid]) / mid
            second_half_avg = sum(scores[mid:]) / (n - mid)
            delta = second_half_avg - first_half_avg
            if delta > 0.03:
                trend_direction = "↑ improving"
                trend_detail = f"+{delta:.3f} avg score (recent half vs earlier half)"
            elif delta < -0.03:
                trend_direction = "↓ declining"
                trend_detail = f"{delta:.3f} avg score (recent half vs earlier half)"
            else:
                trend_direction = "→ steady"
                trend_detail = f"Δ{delta:+.3f} avg score (within noise)"

        dim_labels = {
            "semantic_distance": "Semantic Distance (connecting far-apart ideas)",
            "domain_crossings": "Domain Crossings (bridging different fields)",
            "surprise": "Surprise Factor (unexpected connections)",
            "bridgeability": "Bridgeability (how well ideas connect)",
            "novelty": "Novelty (freshness vs. past outputs)",
        }

        return CreativityReport(
            total_interjections=n,
            aha_count=aha_count,
            aha_rate=(aha_count / n) * 100 if n else 0,
            avg_score=sum(scores) / n,
            avg_semantic_distance=avg_dims["semantic_distance"],
            avg_domain_crossings=avg_dims["domain_crossings"],
            avg_surprise=avg_dims["surprise"],
            avg_bridgeability=avg_dims["bridgeability"],
            avg_novelty=avg_dims["novelty"],
            best_score=max(scores),
            best_chain_summary=best_chain.chain_summary[:120],
            remote_association_count=remote_count,
            remote_association_rate=(remote_count / n) * 100 if n else 0,
            user_rated_count=len(rated),
            avg_user_rating=avg_user_rating,
            user_aha_alignment=aha_alignment * 100,
            trend_direction=trend_direction,
            trend_detail=trend_detail,
            strongest_dimension=dim_labels.get(strongest, strongest),
            weakest_dimension=dim_labels.get(weakest, weakest),
            period_label=period_label,
            first_timestamp=snapshots[0].timestamp if snapshots else 0,
            last_timestamp=snapshots[-1].timestamp if snapshots else 0,
        )

    def format_report(self, report: CreativityReport) -> str:
        """Format a CreativityReport into a beautiful terminal display."""
        if report.total_interjections == 0:
            return (
                f"\n{'═' * 70}\n"
                f"📊 CREATIVITY SELF-EVALUATION\n"
                f"{'═' * 70}\n"
                f"   No interjections recorded yet. Fire some heartbeats first! 🚀\n"
                f"{'═' * 70}"
            )

        # Time range display
        from datetime import datetime
        start = datetime.fromtimestamp(report.first_timestamp).strftime("%b %d %H:%M") if report.first_timestamp else "?"
        end = datetime.fromtimestamp(report.last_timestamp).strftime("%b %d %H:%M") if report.last_timestamp else "?"

        lines = [
            f"\n{'═' * 70}",
            f"📊 CREATIVITY SELF-EVALUATION  [{report.period_label}]",
            f"   {start} → {end}",
            f"{'═' * 70}",
            f"",
            f"   🎯 AHA! RATE",
            f"   ┌──────────────────────────────────────────────────┐",
            f"   │  {report.aha_count} AHA! moments out of {report.total_interjections} interjections = {report.aha_rate:.1f}%  │",
            f"   │  {self._aha_bar(report.aha_rate)}  │",
            f"   └──────────────────────────────────────────────────┘",
            f"",
            f"   📈 SCORING AVERAGES (Mednick/Kenett metrics)",
            f"   │  Semantic Distance : {self._bar(report.avg_semantic_distance)} {report.avg_semantic_distance:.3f}",
            f"   │  Domain Crossings  : {self._bar(report.avg_domain_crossings)} {report.avg_domain_crossings:.3f}",
            f"   │  Surprise          : {self._bar(report.avg_surprise)} {report.avg_surprise:.3f}",
            f"   │  Bridgeability     : {self._bar(report.avg_bridgeability)} {report.avg_bridgeability:.3f}",
            f"   │  Novelty           : {self._bar(report.avg_novelty)} {report.avg_novelty:.3f}",
            f"   │  ─────────────────────────────────",
            f"   │  OVERALL AVG       : {self._bar(report.avg_score)} {report.avg_score:.3f}",
            f"   │  BEST EVER         : {report.best_score:.3f} ⭐",
            f"",
            f"   🔬 REMOTE ASSOCIATIONS (Mednick distance)",
            f"   │  {report.remote_association_count}/{report.total_interjections} chains made genuinely remote connections ({report.remote_association_rate:.1f}%)",
            f"",
            f"   💪 STRONGEST: {report.strongest_dimension}",
            f"   🎯 WEAKEST:  {report.weakest_dimension}",
            f"",
        ]

        if report.user_rated_count > 0:
            lines.extend([
                f"   👤 USER FEEDBACK ({report.user_rated_count} rated)",
                f"   │  Avg rating: {'⭐' * round(report.avg_user_rating)} ({report.avg_user_rating:.1f}/5)",
                f"   │  AHA-alignment: {report.user_aha_alignment:.0f}% of high-score outputs also got high ratings",
                f"",
            ])

        lines.extend([
            f"   📉 TREND: {report.trend_direction}",
            f"   │  {report.trend_detail}" if report.trend_detail else "",
            f"",
            f"   🏆 BEST CHAIN: \"{report.best_chain_summary}\"",
            f"{'═' * 70}",
        ])

        return "\n".join(line for line in lines if line is not None)

    @staticmethod
    def _bar(value: float, width: int = 20) -> str:
        """Render a horizontal bar chart character."""
        filled = int(value * width)
        return "█" * filled + "░" * (width - filled)

    @staticmethod
    def _aha_bar(rate: float) -> str:
        """Render the AHA! rate as a visual progress bar."""
        filled = int(rate / 5)  # 20 chars = 100%
        bar = "🟢" * min(filled, 20) + "⚫" * max(20 - filled, 0)
        if rate >= 50:
            label = "🔥 On fire!"
        elif rate >= 30:
            label = "✨ Solid creativity"
        elif rate >= 15:
            label = "🌱 Growing"
        else:
            label = "🧊 Warming up"
        return f"{bar} {label}"
