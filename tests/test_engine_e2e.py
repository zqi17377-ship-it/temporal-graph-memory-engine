"""End-to-end tests for the 3-Tier HierarchicalGraphMemoryEngine, WorkingMemory, EpisodicStore, and Background Write Pipeline."""

from __future__ import annotations

from pathlib import Path

from graph_memory import (
    DialogueTurn,
    EpisodicVectorStore,
    HierarchicalGraphMemoryEngine,
    QueryIntent,
    WorkingMemory,
)


def test_working_memory_salience_gated_eviction() -> None:
    wm = WorkingMemory(capacity=2, salience_threshold=0.35)

    t1_salient = DialogueTurn(
        turn_id=1,
        session_id="s1",
        user_id="u1",
        content="I live in Beijing and switched my primary database to PostgreSQL.",
    )
    t2_noise = DialogueTurn(
        turn_id=2,
        session_id="s1",
        user_id="u1",
        content="Haha rainy weather today, tired after coffee.",
    )
    t3_next = DialogueTurn(
        turn_id=3,
        session_id="s1",
        user_id="u1",
        content="I started leading Project Helios.",
    )
    t4_next = DialogueTurn(
        turn_id=4,
        session_id="s1",
        user_id="u1",
        content="Project Helios collaborates with Dr. Mei Zhou.",
    )

    assert wm.push(t1_salient) is None
    assert wm.push(t2_noise) is None
    ev1 = wm.push(t3_next)
    assert ev1 is not None and ev1.turn_id == 1
    ev2 = wm.push(t4_next)
    assert ev2 is None

    assert wm.evicted_promoted_count == 1
    assert wm.evicted_discarded_count == 1


def test_episodic_vector_store_persistence_and_cosine_search(tmp_path: Path) -> None:
    ep_file = tmp_path / "episodes.sqlite3"
    store1 = EpisodicVectorStore(ep_file)
    store1.initialize_sync()

    store1.add_turn_as_episode_sync(
        DialogueTurn(
            turn_id=1,
            session_id="s1",
            user_id="u1",
            content="We migrated our primary database engine to PostgreSQL for ACID transactions.",
            salience_score=0.9,
        ),
        episode_id="ep_db",
    )
    store1.add_turn_as_episode_sync(
        DialogueTurn(
            turn_id=2,
            session_id="s1",
            user_id="u1",
            content="My dietary preference is Vegetarian when ordering team meals.",
            salience_score=0.88,
        ),
        episode_id="ep_diet",
    )
    store1.close_sync()

    # Reload from SQLite file on disk and verify normalized vector cosine search
    store2 = EpisodicVectorStore(ep_file)
    store2.initialize_sync()
    assert store2.count("u1") == 2

    hits = store2.search_sync("Which primary database engine do we use?", user_id="u1", top_k=2)
    assert len(hits) >= 1
    assert hits[0][0].episode_id == "ep_db"
    store2.close_sync()


def test_hybrid_engine_conflict_override_and_historical_routing() -> None:
    engine = HierarchicalGraphMemoryEngine(db_dir=":memory:", working_capacity=3)
    engine.register_user("u_alex", "Alex Chen")

    turns = [
        DialogueTurn(turn_id=1, session_id="s1", user_id="u_alex", content="Currently I live in Beijing."),
        DialogueTurn(turn_id=2, session_id="s1", user_id="u_alex", content="My primary database is MySQL."),
        DialogueTurn(turn_id=10, session_id="s2", user_id="u_alex", content="I relocated to Hangzhou this month."),
        DialogueTurn(turn_id=18, session_id="s3", user_id="u_alex", content="We switched our primary database from MySQL to PostgreSQL."),
        DialogueTurn(turn_id=25, session_id="s4", user_id="u_alex", content="Haha grabbing coffee on a rainy afternoon."),
    ]
    for t in turns:
        engine.ingest_turn_sync(t)
    engine.flush_working_memory_sync("u_alex")

    ans_city, res_city = engine.answer_query_sync(
        "u_alex", "Which city does Alex Chen currently live in?", current_turn=30
    )
    assert res_city.intent == QueryIntent.CURRENT_STATE
    assert ans_city == "Hangzhou"
    assert "Beijing" not in res_city.formatted_prompt_context

    ans_db, _ = engine.answer_query_sync(
        "u_alex", "What is Alex Chen's primary database right now?", current_turn=30
    )
    assert ans_db == "PostgreSQL"

    ans_hist, res_hist = engine.answer_query_sync(
        "u_alex", "Which city did Alex Chen live in before moving to Hangzhou?", current_turn=30
    )
    assert res_hist.intent == QueryIntent.HISTORICAL_LINEAGE
    assert ans_hist == "Beijing"
    engine.close_sync()


def test_hybrid_engine_three_hop_cross_session_reasoning() -> None:
    engine = HierarchicalGraphMemoryEngine(db_dir=":memory:", working_capacity=4)
    engine.register_user("u_alex", "Alex Chen")

    turns = [
        DialogueTurn(turn_id=2, session_id="s1", user_id="u_alex", content="I started leading Project Helios."),
        DialogueTurn(turn_id=15, session_id="s2", user_id="u_alex", content="For Project Helios, I teamed up with Dr. Mei Zhou."),
        DialogueTurn(turn_id=34, session_id="s4", user_id="u_alex", content="Dr. Mei Zhou works at Tsinghua AIR."),
    ]
    for t in turns:
        engine.ingest_turn_sync(t)

    ans, res = engine.answer_query_sync(
        "u_alex",
        "Which organization is the collaborator on the project led by Alex Chen affiliated with?",
        current_turn=40,
    )
    assert res.intent == QueryIntent.MULTI_HOP_RELATIONAL
    assert ans == "Tsinghua AIR"
    engine.close_sync()


def test_background_write_queue_pipeline_and_disk_persistence(tmp_path: Path) -> None:
    db_dir = tmp_path / "bg_engine_db"
    engine = HierarchicalGraphMemoryEngine(db_dir=db_dir, working_capacity=2)
    engine.register_user("u_bg", "Lin Wei")

    engine.enqueue_turn_bg(
        DialogueTurn(turn_id=1, session_id="s1", user_id="u_bg", content="My GPU budget is Single RTX 3090 (24GB).")
    )
    engine.enqueue_turn_bg(
        DialogueTurn(turn_id=12, session_id="s2", user_id="u_bg", content="Our GPU budget was upgraded to 4x A100 (320GB).")
    )
    engine.drain_background_queue_sync()
    engine.flush_working_memory_sync("u_bg")
    engine.close_sync()

    # Recover full engine from SQLite directory and query
    recovered = HierarchicalGraphMemoryEngine(db_dir=db_dir, working_capacity=2)
    recovered.register_user("u_bg", "Lin Wei")
    recovered.initialize_sync()
    ans, _ = recovered.answer_query_sync(
        "u_bg", "What is Lin Wei's current GPU budget?", current_turn=15
    )
    assert ans == "4x A100 (320GB)"
    recovered.close_sync()
