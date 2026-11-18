"""Fact Override & Conflict Resolution Engine for Temporal Knowledge Graphs."""

from __future__ import annotations

import uuid

from graph_memory.models import (
    ConflictResolutionEvent,
    EntityNode,
    ExtractedFact,
    PredicateCategory,
    TemporalEdge,
)
from graph_memory.temporal_graph import TemporalKnowledgeGraph

FUNCTIONAL_PREDICATES: set[str] = {
    "lives_in",
    "works_at",
    "current_role",
    "primary_db",
    "target_conference",
    "dietary_preference",
    "gpu_budget",
    "advisor",
    "os_preference",
    "deployment_cloud",
}

ADDITIVE_PREDICATES: set[str] = {
    "speaks_language",
    "completed_course",
    "collaborates_with",
    "uses_library",
    "works_on",
    "research_interest",
    "mentioned_topic",
    "ate_meal",
    "casual_activity",
}

RETRACTION_TARGET_MAP: dict[str, list[str]] = {
    "no_longer_uses": ["uses_library", "primary_db"],
    "left_project": ["works_on"],
    "no_longer_collaborates_with": ["collaborates_with"],
}


class ConflictResolver:
    """Resolves functional overrides, additive coexistence, duplicate reinforcement, and retractions."""

    def __init__(
        self,
        functional_predicates: set[str] | None = None,
        additive_predicates: set[str] | None = None,
    ) -> None:
        self.functional_predicates = (
            set(functional_predicates)
            if functional_predicates is not None
            else set(FUNCTIONAL_PREDICATES)
        )
        self.additive_predicates = (
            set(additive_predicates)
            if additive_predicates is not None
            else set(ADDITIVE_PREDICATES)
        )
        self.audit_log: list[ConflictResolutionEvent] = []

    def classify_predicate(self, predicate: str, is_retraction: bool = False) -> PredicateCategory:
        norm = predicate.strip().lower()
        if is_retraction or norm in RETRACTION_TARGET_MAP:
            return PredicateCategory.RETRACTION
        if norm in self.functional_predicates:
            return PredicateCategory.FUNCTIONAL
        return PredicateCategory.ADDITIVE

    def _compute_resolution(
        self,
        graph: TemporalKnowledgeGraph,
        subj_node: EntityNode,
        obj_node: EntityNode,
        fact: ExtractedFact,
        user_id: str,
        current_turn: int,
        source_episode_id: str,
    ) -> tuple[TemporalEdge | None, list[TemporalEdge], ConflictResolutionEvent]:
        """Compute edges to upsert (closed + new/reinforced) and audit event."""
        predicate = fact.predicate.strip().lower()
        category = self.classify_predicate(predicate, fact.is_retraction)
        edges_to_write: list[TemporalEdge] = []

        if category == PredicateCategory.RETRACTION:
            target_preds = (
                [fact.target_predicate]
                if fact.target_predicate
                else RETRACTION_TARGET_MAP.get(predicate, ["uses_library"])
            )
            closed_ids: list[str] = []
            retraction_marker = f"retracted_at_turn_{current_turn}"

            for t_pred in target_preds:
                matching_active = graph.get_active_edges(
                    user_id=user_id,
                    subject=subj_node.node_id,
                    predicate=t_pred,
                    object_id=obj_node.node_id,
                )
                for old_edge in matching_active:
                    closed = old_edge.model_copy(
                        update={
                            "valid_until_turn": current_turn,
                            "superseded_by": retraction_marker,
                        }
                    )
                    edges_to_write.append(closed)
                    closed_ids.append(old_edge.edge_id)

            event = ConflictResolutionEvent(
                action="RETRACTED",
                turn=current_turn,
                subject=subj_node.node_id,
                predicate=predicate,
                new_object=obj_node.node_id,
                old_object=obj_node.node_id if closed_ids else None,
                new_edge_id=None,
                closed_edge_ids=closed_ids,
            )
            return None, edges_to_write, event

        if category == PredicateCategory.FUNCTIONAL:
            active_for_slot = graph.get_active_edges(
                user_id=user_id,
                subject=subj_node.node_id,
                predicate=predicate,
            )
            for existing in active_for_slot:
                if existing.object == obj_node.node_id:
                    reinforced = existing.model_copy(
                        update={
                            "access_count": existing.access_count + 1,
                            "last_accessed_turn": max(existing.last_accessed_turn, current_turn),
                            "confidence": max(existing.confidence, fact.confidence),
                        }
                    )
                    edges_to_write.append(reinforced)
                    event = ConflictResolutionEvent(
                        action="REINFORCED",
                        turn=current_turn,
                        subject=subj_node.node_id,
                        predicate=predicate,
                        new_object=obj_node.node_id,
                        old_object=obj_node.node_id,
                        new_edge_id=reinforced.edge_id,
                        closed_edge_ids=[],
                    )
                    return reinforced, edges_to_write, event

            new_edge_id = f"edge_{predicate}_{current_turn:04d}_{uuid.uuid4().hex[:6]}"
            new_edge = TemporalEdge(
                edge_id=new_edge_id,
                user_id=user_id,
                subject=subj_node.node_id,
                predicate=predicate,
                object=obj_node.node_id,
                valid_from_turn=current_turn,
                valid_until_turn=None,
                confidence=fact.confidence,
                base_importance=fact.base_importance,
                access_count=0,
                last_accessed_turn=current_turn,
                source_episode_id=source_episode_id,
                superseded_by=None,
                archived=False,
            )

            closed_ids = []
            old_obj: str | None = None
            for old_edge in active_for_slot:
                old_obj = old_edge.object
                closed_old = old_edge.model_copy(
                    update={
                        "valid_until_turn": current_turn,
                        "superseded_by": new_edge_id,
                    }
                )
                edges_to_write.append(closed_old)
                closed_ids.append(old_edge.edge_id)

            edges_to_write.append(new_edge)
            action = "SUPERSEDED" if closed_ids else "INSERTED"
            event = ConflictResolutionEvent(
                action=action,
                turn=current_turn,
                subject=subj_node.node_id,
                predicate=predicate,
                new_object=obj_node.node_id,
                old_object=old_obj,
                new_edge_id=new_edge_id,
                closed_edge_ids=closed_ids,
            )
            return new_edge, edges_to_write, event

        # Additive predicate
        existing_same = graph.get_active_edges(
            user_id=user_id,
            subject=subj_node.node_id,
            predicate=predicate,
            object_id=obj_node.node_id,
        )
        if existing_same:
            target = existing_same[0]
            reinforced = target.model_copy(
                update={
                    "access_count": target.access_count + 1,
                    "last_accessed_turn": max(target.last_accessed_turn, current_turn),
                    "confidence": max(target.confidence, fact.confidence),
                }
            )
            edges_to_write.append(reinforced)
            event = ConflictResolutionEvent(
                action="REINFORCED",
                turn=current_turn,
                subject=subj_node.node_id,
                predicate=predicate,
                new_object=obj_node.node_id,
                old_object=obj_node.node_id,
                new_edge_id=reinforced.edge_id,
                closed_edge_ids=[],
            )
            return reinforced, edges_to_write, event

        coexisting = graph.get_active_edges(
            user_id=user_id,
            subject=subj_node.node_id,
            predicate=predicate,
        )
        new_edge_id = f"edge_{predicate}_{current_turn:04d}_{uuid.uuid4().hex[:6]}"
        new_edge = TemporalEdge(
            edge_id=new_edge_id,
            user_id=user_id,
            subject=subj_node.node_id,
            predicate=predicate,
            object=obj_node.node_id,
            valid_from_turn=current_turn,
            valid_until_turn=None,
            confidence=fact.confidence,
            base_importance=fact.base_importance,
            access_count=0,
            last_accessed_turn=current_turn,
            source_episode_id=source_episode_id,
            superseded_by=None,
            archived=False,
        )
        edges_to_write.append(new_edge)
        action = "ADDED_COEXISTING" if coexisting else "INSERTED"
        event = ConflictResolutionEvent(
            action=action,
            turn=current_turn,
            subject=subj_node.node_id,
            predicate=predicate,
            new_object=obj_node.node_id,
            old_object=coexisting[-1].object if coexisting else None,
            new_edge_id=new_edge_id,
            closed_edge_ids=[],
        )
        return new_edge, edges_to_write, event

    def apply_fact_sync(
        self,
        graph: TemporalKnowledgeGraph,
        fact: ExtractedFact,
        user_id: str,
        current_turn: int,
        source_episode_id: str = "",
    ) -> tuple[TemporalEdge | None, ConflictResolutionEvent]:
        """Synchronously apply an ExtractedFact with validity-interval conflict resolution."""
        subj_node = graph.upsert_node_sync(fact.subject, fact.subject_type, current_turn)
        obj_node = graph.upsert_node_sync(fact.object, fact.object_type, current_turn)
        primary_edge, to_write, event = self._compute_resolution(
            graph, subj_node, obj_node, fact, user_id, current_turn, source_episode_id
        )
        for e in to_write:
            graph.upsert_edge_sync(e)
        self.audit_log.append(event)
        return primary_edge, event

    async def apply_fact(
        self,
        graph: TemporalKnowledgeGraph,
        fact: ExtractedFact,
        user_id: str,
        current_turn: int,
        source_episode_id: str = "",
    ) -> tuple[TemporalEdge | None, ConflictResolutionEvent]:
        """Asynchronously apply an ExtractedFact with validity-interval conflict resolution."""
        subj_node = await graph.upsert_node(fact.subject, fact.subject_type, current_turn)
        obj_node = await graph.upsert_node(fact.object, fact.object_type, current_turn)
        primary_edge, to_write, event = self._compute_resolution(
            graph, subj_node, obj_node, fact, user_id, current_turn, source_episode_id
        )
        for e in to_write:
            await graph.upsert_edge(e)
        self.audit_log.append(event)
        return primary_edge, event
