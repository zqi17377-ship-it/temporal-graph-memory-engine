"""Tests for Ebbinghaus Time-Decay Forgetting Math, Reinforcement, and Consolidation Pruning."""

from __future__ import annotations

import math

from graph_memory import (
    ConflictResolver,
    EntityType,
    ExtractedFact,
    ForgettingEngine,
    TemporalEdge,
    TemporalKnowledgeGraph,
)


def test_ebbinghaus_formula_exact_value() -> None:
    fe = ForgettingEngine(decay_lambda=0.02, reinforcement_alpha=1.5, consolidation_theta=0.25)
    edge = TemporalEdge(
        edge_id="e_math",
        subject="person:u1",
        predicate="lives_in",
        object="city:hangzhou",
        valid_from_turn=5,
        base_importance=0.90,
        access_count=3,
        last_accessed_turn=10,
    )
    expected = 0.90 * math.exp(-(0.02 * 25) / (1.0 + 1.5 * math.log(1.0 + 3)))
    actual = fe.retention_score(edge, current_turn=35)
    assert math.isclose(actual, expected, rel_tol=1e-9)


def test_access_reinforcement_flattens_decay_curve() -> None:
    fe = ForgettingEngine(decay_lambda=0.025, reinforcement_alpha=1.8, consolidation_theta=0.35)
    unreinforced = TemporalEdge(
        edge_id="e_cold",
        subject="person:u1",
        predicate="works_on",
        object="project:alpha",
        valid_from_turn=1,
        base_importance=0.85,
        access_count=0,
        last_accessed_turn=1,
    )
    reinforced = unreinforced.model_copy(update={"edge_id": "e_warm", "access_count": 5})

    s_cold = fe.retention_score(unreinforced, current_turn=45)
    s_warm = fe.retention_score(reinforced, current_turn=45)

    assert s_warm > s_cold
    assert fe.should_prune(unreinforced, current_turn=45, theta=0.35) is True
    assert fe.should_prune(reinforced, current_turn=45, theta=0.35) is False


def test_consolidation_prunes_stale_chatter_while_preserving_core_facts() -> None:
    graph = TemporalKnowledgeGraph(":memory:")
    graph.initialize_sync()
    resolver = ConflictResolver()
    fe = ForgettingEngine(decay_lambda=0.016, reinforcement_alpha=1.8, consolidation_theta=0.25)

    resolver.apply_fact_sync(
        graph,
        ExtractedFact(
            subject="Alex Chen",
            subject_type=EntityType.PERSON,
            predicate="advisor",
            object="Prof. Hao Zhang",
            object_type=EntityType.PERSON,
            base_importance=0.90,
        ),
        user_id="u1",
        current_turn=4,
    )
    resolver.apply_fact_sync(
        graph,
        ExtractedFact(
            subject="Alex Chen",
            subject_type=EntityType.PERSON,
            predicate="ate_meal",
            object="Spicy Ramen",
            object_type=EntityType.PREFERENCE,
            base_importance=0.30,
        ),
        user_id="u1",
        current_turn=4,
    )

    report = fe.consolidate_sync(graph, current_turn=50, user_id="u1", theta=0.25)
    assert report.total_active_before == 2
    assert report.archived_count == 1
    assert report.retained_count == 1
    assert math.isclose(report.compression_ratio, 0.5, rel_tol=1e-4)

    active = graph.get_active_edges(user_id="u1")
    assert len(active) == 1
    assert active[0].predicate == "advisor"
    graph.close_sync()


def test_theta_zero_never_prunes_any_active_edges() -> None:
    graph = TemporalKnowledgeGraph(":memory:")
    graph.initialize_sync()
    resolver = ConflictResolver()
    fe = ForgettingEngine(decay_lambda=0.03, reinforcement_alpha=1.5, consolidation_theta=0.0)

    resolver.apply_fact_sync(
        graph,
        ExtractedFact(
            subject="Alex Chen",
            subject_type=EntityType.PERSON,
            predicate="mentioned_topic",
            object="Rainy Weather",
            object_type=EntityType.PREFERENCE,
            base_importance=0.20,
        ),
        user_id="u1",
        current_turn=1,
    )

    report = fe.consolidate_sync(graph, current_turn=100, user_id="u1", theta=0.0)
    assert report.archived_count == 0
    assert report.compression_ratio == 0.0
    assert len(graph.get_active_edges(user_id="u1")) == 1
    graph.close_sync()
