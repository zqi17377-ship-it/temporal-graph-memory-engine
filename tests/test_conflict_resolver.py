"""Tests for Functional vs Additive Conflict Resolution and Explicit Retraction."""

from __future__ import annotations

from collections.abc import Coroutine
from typing import TypeVar

from graph_memory import (
    ConflictResolver,
    EntityType,
    ExtractedFact,
    PredicateCategory,
    TemporalKnowledgeGraph,
)

T = TypeVar("T")


def run_in_memory_coro(coro: Coroutine[object, object, T]) -> T:
    """Drive a pure in-memory coroutine to completion without opening OS loopback sockets."""
    try:
        while True:
            coro.send(None)
    except StopIteration as stop:
        return stop.value


def test_functional_predicate_supersedes_and_preserves_lineage() -> None:
    async def _scenario() -> None:
        graph = TemporalKnowledgeGraph(":memory:")
        await graph.initialize()
        resolver = ConflictResolver()

        e1, ev1 = await resolver.apply_fact(
            graph,
            ExtractedFact(
                subject="Alex Chen",
                subject_type=EntityType.PERSON,
                predicate="lives_in",
                object="Beijing",
                object_type=EntityType.CITY,
            ),
            user_id="u1",
            current_turn=2,
            source_episode_id="ep_0002",
        )
        assert ev1.action == "INSERTED"
        assert e1 is not None and e1.is_active

        e2, ev2 = await resolver.apply_fact(
            graph,
            ExtractedFact(
                subject="Alex Chen",
                subject_type=EntityType.PERSON,
                predicate="lives_in",
                object="Hangzhou",
                object_type=EntityType.CITY,
            ),
            user_id="u1",
            current_turn=15,
            source_episode_id="ep_0015",
        )
        assert ev2.action == "SUPERSEDED"
        assert e2 is not None and e2.is_active
        assert ev2.closed_edge_ids == [e1.edge_id]

        active = graph.get_active_edges(user_id="u1", predicate="lives_in")
        assert len(active) == 1
        assert graph.display_name(active[0].object) == "Hangzhou"

        lineage = graph.get_predicate_lineage("Alex Chen", "lives_in", user_id="u1")
        assert len(lineage) == 2
        assert lineage[0].valid_from_turn == 2
        assert lineage[0].valid_until_turn == 15
        assert lineage[0].superseded_by == e2.edge_id
        assert lineage[1].valid_from_turn == 15
        assert lineage[1].valid_until_turn is None
        await graph.close()

    run_in_memory_coro(_scenario())


def test_additive_predicates_coexist_without_closing_prior_edges() -> None:
    graph = TemporalKnowledgeGraph(":memory:")
    graph.initialize_sync()
    resolver = ConflictResolver()

    for turn_idx, lang in [(3, "Mandarin"), (9, "English"), (21, "Japanese")]:
        _, ev = resolver.apply_fact_sync(
            graph,
            ExtractedFact(
                subject="Alex Chen",
                subject_type=EntityType.PERSON,
                predicate="speaks_language",
                object=lang,
                object_type=EntityType.PREFERENCE,
            ),
            user_id="u1",
            current_turn=turn_idx,
        )
        if turn_idx == 3:
            assert ev.action == "INSERTED"
        else:
            assert ev.action == "ADDED_COEXISTING"

    active_langs = {
        graph.display_name(e.object)
        for e in graph.get_active_edges(user_id="u1", predicate="speaks_language")
    }
    assert active_langs == {"Mandarin", "English", "Japanese"}
    graph.close_sync()


def test_duplicate_fact_reinforces_instead_of_duplicating() -> None:
    graph = TemporalKnowledgeGraph(":memory:")
    graph.initialize_sync()
    resolver = ConflictResolver()

    fact = ExtractedFact(
        subject="Project Helios",
        subject_type=EntityType.PROJECT,
        predicate="primary_db",
        object="psql",  # Alias for PostgreSQL
        object_type=EntityType.TECH_STACK,
    )
    e1, _ = resolver.apply_fact_sync(graph, fact, user_id="u1", current_turn=4)
    assert e1 is not None and e1.access_count == 0

    fact_canon = fact.model_copy(update={"object": "PostgreSQL"})
    e2, ev2 = resolver.apply_fact_sync(graph, fact_canon, user_id="u1", current_turn=18)
    assert ev2.action == "REINFORCED"
    assert e2 is not None
    assert e2.edge_id == e1.edge_id
    assert e2.access_count == 1
    assert e2.last_accessed_turn == 18
    assert len(graph.get_active_edges(user_id="u1", predicate="primary_db")) == 1
    graph.close_sync()


def test_explicit_negation_retraction_closes_target_additive_edge() -> None:
    graph = TemporalKnowledgeGraph(":memory:")
    graph.initialize_sync()
    resolver = ConflictResolver()

    for turn_idx, lib in [(5, "PyTorch"), (6, "LangChain")]:
        resolver.apply_fact_sync(
            graph,
            ExtractedFact(
                subject="Alex Chen",
                subject_type=EntityType.PERSON,
                predicate="uses_library",
                object=lib,
                object_type=EntityType.TECH_STACK,
            ),
            user_id="u1",
            current_turn=turn_idx,
        )

    assert len(graph.get_active_edges(user_id="u1", predicate="uses_library")) == 2

    retracted_edge, ev_ret = resolver.apply_fact_sync(
        graph,
        ExtractedFact(
            subject="Alex Chen",
            subject_type=EntityType.PERSON,
            predicate="no_longer_uses",
            object="LangChain",
            object_type=EntityType.TECH_STACK,
            is_retraction=True,
            target_predicate="uses_library",
        ),
        user_id="u1",
        current_turn=20,
    )
    assert retracted_edge is None
    assert ev_ret.action == "RETRACTED"
    assert len(ev_ret.closed_edge_ids) == 1

    remaining = [
        graph.display_name(e.object)
        for e in graph.get_active_edges(user_id="u1", predicate="uses_library")
    ]
    assert remaining == ["PyTorch"]
    graph.close_sync()


def test_predicate_classification_categories() -> None:
    resolver = ConflictResolver()
    assert resolver.classify_predicate("lives_in") == PredicateCategory.FUNCTIONAL
    assert resolver.classify_predicate("gpu_budget") == PredicateCategory.FUNCTIONAL
    assert resolver.classify_predicate("collaborates_with") == PredicateCategory.ADDITIVE
    assert resolver.classify_predicate("completed_course") == PredicateCategory.ADDITIVE
    assert resolver.classify_predicate("no_longer_uses") == PredicateCategory.RETRACTION
