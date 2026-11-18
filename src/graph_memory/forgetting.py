"""Ebbinghaus Time-Decay Forgetting & Memory Consolidation Engine."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from graph_memory.models import ConsolidationReport, TemporalEdge

if TYPE_CHECKING:
    from graph_memory.temporal_graph import TemporalKnowledgeGraph


class ForgettingEngine:
    r"""Dynamic memory retention & periodic consolidation based on Ebbinghaus spacing law:

    S(\Delta t) = I_0 * exp( - (\lambda * \Delta t) / (1 + \alpha * ln(1 + c_access)) )
    """

    def __init__(
        self,
        decay_lambda: float = 0.016,
        reinforcement_alpha: float = 1.8,
        consolidation_theta: float = 0.25,
    ) -> None:
        if decay_lambda < 0.0:
            raise ValueError("decay_lambda must be >= 0")
        if reinforcement_alpha < 0.0:
            raise ValueError("reinforcement_alpha must be >= 0")
        self.decay_lambda = decay_lambda
        self.reinforcement_alpha = reinforcement_alpha
        self.consolidation_theta = consolidation_theta

    def retention_score(self, edge: TemporalEdge, current_turn: int) -> float:
        """Compute dynamic Ebbinghaus retention score S(Delta t) in [0, 1]."""
        delta_t = max(0, current_turn - edge.last_accessed_turn)
        c_access = max(0, edge.access_count)
        stability = 1.0 + self.reinforcement_alpha * math.log(1.0 + c_access)
        exponent = -(self.decay_lambda * delta_t) / stability
        score = edge.base_importance * math.exp(exponent)
        return float(max(0.0, min(1.0, score)))

    def should_prune(
        self,
        edge: TemporalEdge,
        current_turn: int,
        theta: float | None = None,
    ) -> bool:
        """Check whether an active edge falls strictly below pruning threshold theta."""
        threshold = self.consolidation_theta if theta is None else theta
        if threshold <= 0.0:
            return False
        return self.retention_score(edge, current_turn) < threshold

    def _plan_consolidation(
        self,
        graph: TemporalKnowledgeGraph,
        current_turn: int,
        user_id: str | None = None,
        theta: float | None = None,
    ) -> tuple[float, list[TemporalEdge], list[TemporalEdge], list[str]]:
        threshold = self.consolidation_theta if theta is None else theta
        active_edges = graph.get_active_edges(user_id=user_id)
        previously_archived = [
            e
            for e in graph.get_all_edges(user_id=user_id)
            if e.valid_until_turn is None and e.archived
        ]
        to_archive: list[str] = []
        if threshold > 0.0:
            for edge in active_edges:
                if self.retention_score(edge, current_turn) < threshold:
                    to_archive.append(edge.edge_id)
        return threshold, active_edges, previously_archived, to_archive

    def _build_report(
        self,
        current_turn: int,
        threshold: float,
        active_edges: list[TemporalEdge],
        previously_archived: list[TemporalEdge],
        to_archive: list[str],
    ) -> ConsolidationReport:
        total_pool = len(active_edges) + len(previously_archived)
        cumulative_archived = len(previously_archived) + len(to_archive)
        retained = len(active_edges) - len(to_archive)
        compression_ratio = (
            round(cumulative_archived / total_pool, 4) if total_pool > 0 else 0.0
        )
        return ConsolidationReport(
            current_turn=current_turn,
            threshold_theta=threshold,
            total_active_before=len(active_edges),
            archived_count=len(to_archive),
            retained_count=retained,
            compression_ratio=compression_ratio,
            archived_edge_ids=to_archive,
        )

    def consolidate_sync(
        self,
        graph: TemporalKnowledgeGraph,
        current_turn: int,
        user_id: str | None = None,
        theta: float | None = None,
    ) -> ConsolidationReport:
        """Synchronously archive stale active edges below threshold theta."""
        threshold, active_edges, prev_arch, to_archive = self._plan_consolidation(
            graph, current_turn, user_id, theta
        )
        if to_archive:
            graph.archive_edges_sync(to_archive)
        return self._build_report(current_turn, threshold, active_edges, prev_arch, to_archive)

    async def consolidate(
        self,
        graph: TemporalKnowledgeGraph,
        current_turn: int,
        user_id: str | None = None,
        theta: float | None = None,
    ) -> ConsolidationReport:
        """Asynchronously archive stale active edges below threshold theta."""
        threshold, active_edges, prev_arch, to_archive = self._plan_consolidation(
            graph, current_turn, user_id, theta
        )
        if to_archive:
            await graph.archive_edges(to_archive)
        return self._build_report(current_turn, threshold, active_edges, prev_arch, to_archive)
