"""Hybrid Temporal-Aware Retriever & Async/Background Write Pipeline (3-Tier Memory Engine)."""

from __future__ import annotations

import asyncio
from pathlib import Path
import queue
import re
import threading
import time

from graph_memory.conflict_resolver import ConflictResolver
from graph_memory.embeddings import EmbeddingService
from graph_memory.episodic_store import EpisodicVectorStore
from graph_memory.extractor import ContextGroundedReader, MemoryFactExtractor
from graph_memory.forgetting import ForgettingEngine
from graph_memory.models import (
    ConflictResolutionEvent,
    ConsolidationReport,
    DialogueTurn,
    EntityType,
    QueryIntent,
    RetrievalResult,
    TemporalEdge,
)
from graph_memory.temporal_graph import TemporalKnowledgeGraph
from graph_memory.working_memory import WorkingMemory

_HISTORICAL_QUERY_RE = re.compile(
    r"\b(before|previously|prior to|used to|initially|originally|first|earlier|former|"
    r"history|evolution|evolve|timeline|transition|sequence|how did .+ change)\b|"
    r"(之前|以前|最早|最初|演变|变化|历史)",
    re.IGNORECASE,
)

_MULTIHOP_QUERY_RE = re.compile(
    r"\b(collaborator|teammate|partner|project led by|project maintained by|"
    r"project built by|affiliated with|whose|associated with|through|"
    r"which database does the project|where does .+ collaborator|"
    r"which conference does .+ collaborator|deployed on for the project)\b|"
    r"(合作者|所在项目|关联|所属机构)",
    re.IGNORECASE,
)


def estimate_tokens(text: str) -> int:
    if not text.strip():
        return 0
    return max(1, int(len(text.split()) * 1.35))


class HierarchicalGraphMemoryEngine:
    """Three-Tier Hierarchical & Temporal Graph Long-Term Memory Engine."""

    def __init__(
        self,
        db_dir: str | Path | None = None,
        working_capacity: int = 6,
        salience_threshold: float = 0.35,
        decay_lambda: float = 0.016,
        reinforcement_alpha: float = 1.8,
        consolidation_theta: float = 0.25,
        consolidate_every_n_turns: int | None = None,
        use_openai: bool = False,
        embedder: EmbeddingService | None = None,
    ) -> None:
        self.embedder = embedder or EmbeddingService(use_openai=use_openai)
        if db_dir is None or str(db_dir) == ":memory:":
            ep_path: str | Path = ":memory:"
            gr_path: str | Path = ":memory:"
        else:
            base = Path(db_dir)
            base.mkdir(parents=True, exist_ok=True)
            ep_path = base / "episodic_vectors.sqlite3"
            gr_path = base / "temporal_graph.sqlite3"

        self.working_memory = WorkingMemory(
            capacity=working_capacity,
            salience_threshold=salience_threshold,
        )
        self.episodic_store = EpisodicVectorStore(db_path=ep_path, embedder=self.embedder)
        self.temporal_graph = TemporalKnowledgeGraph(db_path=gr_path)
        self.conflict_resolver = ConflictResolver()
        self.forgetting_engine = ForgettingEngine(
            decay_lambda=decay_lambda,
            reinforcement_alpha=reinforcement_alpha,
            consolidation_theta=consolidation_theta,
        )
        self.extractor = MemoryFactExtractor(use_openai=use_openai)
        self.reader = ContextGroundedReader(extractor=self.extractor, use_openai=use_openai)
        self.consolidate_every_n_turns = consolidate_every_n_turns

        self._user_display_names: dict[str, str] = {}
        self._ingest_count: int = 0
        self._lock = threading.RLock()

        # Thread-backed non-blocking background write pipeline (zero TCP loopback sockets)
        self._thread_queue: queue.Queue[tuple[DialogueTurn, str | None] | None] = queue.Queue()
        self._thread_worker: threading.Thread | None = None

        # Asyncio-backed background write pipeline for async event loops
        self._bg_queue: asyncio.Queue[tuple[DialogueTurn, str | None] | None] | None = None
        self._bg_worker_task: asyncio.Task[None] | None = None
        self._initialized = False

    def initialize_sync(self) -> None:
        with self._lock:
            if not self._initialized:
                self.episodic_store.initialize_sync()
                self.temporal_graph.initialize_sync()
                self._initialized = True

    async def initialize(self) -> None:
        if not self._initialized:
            await self.episodic_store.initialize()
            await self.temporal_graph.initialize()
            self._initialized = True

    def register_user(self, user_id: str, display_name: str) -> None:
        self._user_display_names[user_id] = display_name

    def get_user_display_name(self, user_id: str) -> str:
        return self._user_display_names.get(user_id, user_id)

    def classify_query_intent(self, query: str) -> QueryIntent:
        if _HISTORICAL_QUERY_RE.search(query):
            return QueryIntent.HISTORICAL_LINEAGE
        if _MULTIHOP_QUERY_RE.search(query):
            return QueryIntent.MULTI_HOP_RELATIONAL
        return QueryIntent.CURRENT_STATE

    def ingest_turn_sync(
        self,
        turn: DialogueTurn,
        user_display_name: str | None = None,
    ) -> list[ConflictResolutionEvent]:
        """Synchronously ingest a turn across Working, Episodic, and Temporal Graph tiers."""
        with self._lock:
            if not self._initialized:
                self.initialize_sync()

            if user_display_name:
                self._user_display_names[turn.user_id] = user_display_name
            speaker_name = self.get_user_display_name(turn.user_id)

            self.temporal_graph.upsert_node_sync(
                speaker_name,
                EntityType.PERSON,
                turn.turn_id,
                extra_aliases=[turn.user_id],
            )

            facts = self.extractor.extract_offline(turn.content, user_display_name=speaker_name)
            salience = self.working_memory.compute_salience(turn.content, facts)
            scored_turn = turn.model_copy(update={"salience_score": salience})
            episode_id = f"ep_{turn.user_id}_{turn.turn_id:04d}"

            events: list[ConflictResolutionEvent] = []
            for fact in facts:
                _, ev = self.conflict_resolver.apply_fact_sync(
                    graph=self.temporal_graph,
                    fact=fact,
                    user_id=turn.user_id,
                    current_turn=turn.turn_id,
                    source_episode_id=episode_id,
                )
                events.append(ev)

            evicted_turn = self.working_memory.push(scored_turn, facts)
            if evicted_turn is not None:
                ev_ep_id = f"ep_{evicted_turn.user_id}_{evicted_turn.turn_id:04d}"
                self.episodic_store.add_turn_as_episode_sync(evicted_turn, episode_id=ev_ep_id)

            self._ingest_count += 1
            if (
                self.consolidate_every_n_turns
                and self.consolidate_every_n_turns > 0
                and self._ingest_count % self.consolidate_every_n_turns == 0
            ):
                self.consolidate_sync(current_turn=turn.turn_id, user_id=turn.user_id)

            return events

    async def ingest_turn(
        self,
        turn: DialogueTurn,
        user_display_name: str | None = None,
    ) -> list[ConflictResolutionEvent]:
        """Asynchronously ingest a turn across Working, Episodic, and Temporal Graph tiers."""
        if not self._initialized:
            await self.initialize()

        if user_display_name:
            self._user_display_names[turn.user_id] = user_display_name
        speaker_name = self.get_user_display_name(turn.user_id)

        await self.temporal_graph.upsert_node(
            speaker_name,
            EntityType.PERSON,
            turn.turn_id,
            extra_aliases=[turn.user_id],
        )

        facts = await self.extractor.aextract(turn, user_display_name=speaker_name)
        salience = self.working_memory.compute_salience(turn.content, facts)
        scored_turn = turn.model_copy(update={"salience_score": salience})
        episode_id = f"ep_{turn.user_id}_{turn.turn_id:04d}"

        events: list[ConflictResolutionEvent] = []
        for fact in facts:
            _, ev = await self.conflict_resolver.apply_fact(
                graph=self.temporal_graph,
                fact=fact,
                user_id=turn.user_id,
                current_turn=turn.turn_id,
                source_episode_id=episode_id,
            )
            events.append(ev)

        evicted_turn = self.working_memory.push(scored_turn, facts)
        if evicted_turn is not None:
            ev_ep_id = f"ep_{evicted_turn.user_id}_{evicted_turn.turn_id:04d}"
            await self.episodic_store.add_turn_as_episode(evicted_turn, episode_id=ev_ep_id)

        self._ingest_count += 1
        if (
            self.consolidate_every_n_turns
            and self.consolidate_every_n_turns > 0
            and self._ingest_count % self.consolidate_every_n_turns == 0
        ):
            await self.consolidate(current_turn=turn.turn_id, user_id=turn.user_id)

        return events

    def enqueue_turn_bg(
        self,
        turn: DialogueTurn,
        user_display_name: str | None = None,
    ) -> None:
        """Non-blocking background write enqueue via worker thread."""
        if not self._initialized:
            self.initialize_sync()
        if self._thread_worker is None or not self._thread_worker.is_alive():
            self._thread_worker = threading.Thread(
                target=self._thread_worker_loop, daemon=True, name="GraphMemoryWriteWorker"
            )
            self._thread_worker.start()
        self._thread_queue.put((turn, user_display_name))

    def _thread_worker_loop(self) -> None:
        while True:
            item = self._thread_queue.get()
            try:
                if item is None:
                    return
                turn, disp = item
                self.ingest_turn_sync(turn, user_display_name=disp)
            finally:
                self._thread_queue.task_done()

    def drain_background_queue_sync(self) -> None:
        """Block until all background-queued turns finish extraction and persistence."""
        self._thread_queue.join()

    async def enqueue_turn(
        self,
        turn: DialogueTurn,
        user_display_name: str | None = None,
    ) -> None:
        """Asynchronously enqueue a turn for background extraction on an asyncio loop."""
        if not self._initialized:
            await self.initialize()
        if self._bg_queue is None:
            self._bg_queue = asyncio.Queue()
        if self._bg_worker_task is None or self._bg_worker_task.done():
            self._bg_worker_task = asyncio.create_task(self._background_worker())
        await self._bg_queue.put((turn, user_display_name))

    async def _background_worker(self) -> None:
        assert self._bg_queue is not None
        while True:
            item = await self._bg_queue.get()
            try:
                if item is None:
                    return
                turn, disp = item
                await self.ingest_turn(turn, user_display_name=disp)
            finally:
                self._bg_queue.task_done()

    async def drain_background_queue(self) -> None:
        if self._bg_queue is not None:
            await self._bg_queue.join()

    def flush_working_memory_sync(self, user_id: str) -> int:
        with self._lock:
            promoted = self.working_memory.flush_user(user_id)
            for t in promoted:
                ep_id = f"ep_{t.user_id}_{t.turn_id:04d}"
                self.episodic_store.add_turn_as_episode_sync(t, episode_id=ep_id)
            return len(promoted)

    async def flush_working_memory(self, user_id: str) -> int:
        promoted = self.working_memory.flush_user(user_id)
        for t in promoted:
            ep_id = f"ep_{t.user_id}_{t.turn_id:04d}"
            await self.episodic_store.add_turn_as_episode(t, episode_id=ep_id)
        return len(promoted)

    def consolidate_sync(
        self,
        current_turn: int,
        user_id: str | None = None,
        theta: float | None = None,
    ) -> ConsolidationReport:
        with self._lock:
            if not self._initialized:
                self.initialize_sync()
            return self.forgetting_engine.consolidate_sync(
                graph=self.temporal_graph,
                current_turn=current_turn,
                user_id=user_id,
                theta=theta,
            )

    async def consolidate(
        self,
        current_turn: int,
        user_id: str | None = None,
        theta: float | None = None,
    ) -> ConsolidationReport:
        if not self._initialized:
            await self.initialize()
        return await self.forgetting_engine.consolidate(
            graph=self.temporal_graph,
            current_turn=current_turn,
            user_id=user_id,
            theta=theta,
        )

    def _compute_graph_retrieval(
        self,
        user_id: str,
        query: str,
        current_turn: int,
        top_k_graph_edges: int,
        at_turn: int | None,
    ) -> tuple[
        QueryIntent,
        str,
        list[TemporalEdge],
        list[TemporalEdge],
        list[list[TemporalEdge]],
        set[str],
    ]:
        intent = self.classify_query_intent(query)
        speaker_name = self.get_user_display_name(user_id)
        user_node_id = self.temporal_graph.resolve_entity_id(speaker_name) or f"person:{user_id}"

        matched_nodes = self.temporal_graph.match_entities_in_text(query)
        seed_ids = [n.node_id for n in matched_nodes]
        if user_node_id not in seed_ids:
            seed_ids.append(user_node_id)

        target_pred = self.reader._detect_target_predicate(query.lower())

        active_edges: list[TemporalEdge] = []
        historical_edges: list[TemporalEdge] = []
        multi_hop_paths: list[list[TemporalEdge]] = []
        reinforced_ids: set[str] = set()

        if at_turn is not None:
            active_edges = self.temporal_graph.get_edges_at_turn(turn_t=at_turn, user_id=user_id)
        elif intent == QueryIntent.HISTORICAL_LINEAGE:
            candidate_subjects = list(dict.fromkeys(seed_ids))
            for pe in self.temporal_graph.get_active_edges(
                user_id=user_id, subject=user_node_id, predicate="works_on"
            ):
                if pe.object not in candidate_subjects:
                    candidate_subjects.append(pe.object)

            for subj_id in candidate_subjects:
                lineage = self.temporal_graph.get_predicate_lineage(
                    subject=subj_id,
                    predicate=target_pred,
                    user_id=user_id,
                )
                for e in lineage:
                    if e not in historical_edges:
                        historical_edges.append(e)
                        if e.is_active:
                            reinforced_ids.add(e.edge_id)
        elif intent == QueryIntent.MULTI_HOP_RELATIONAL:
            multi_hop_paths = self.temporal_graph.find_multi_hop_paths(
                seed_node_ids=seed_ids,
                max_hops=3,
                user_id=user_id,
                active_only=True,
                exclude_predicates={"ate_meal", "mentioned_topic", "casual_activity"},
            )[:6]
            ppr_edges = self.temporal_graph.personalized_pagerank(
                seed_node_ids=seed_ids,
                current_turn=current_turn,
                user_id=user_id,
                top_k_edges=top_k_graph_edges,
                active_only=True,
                forgetting_engine=self.forgetting_engine,
            )
            active_edges = list(ppr_edges)
            for p in multi_hop_paths:
                for e in p:
                    reinforced_ids.add(e.edge_id)
                    if e not in active_edges:
                        active_edges.append(e)
        else:
            ppr_edges = self.temporal_graph.personalized_pagerank(
                seed_node_ids=seed_ids,
                current_turn=current_turn,
                user_id=user_id,
                top_k_edges=top_k_graph_edges,
                active_only=True,
                forgetting_engine=self.forgetting_engine,
            )
            direct_matches: list[TemporalEdge] = []
            if target_pred:
                for sid in seed_ids:
                    direct_matches.extend(
                        self.temporal_graph.get_active_edges(
                            user_id=user_id, subject=sid, predicate=target_pred
                        )
                    )
                if not direct_matches:
                    direct_matches.extend(
                        self.temporal_graph.get_active_edges(
                            user_id=user_id, predicate=target_pred
                        )
                    )

            seen_eids: set[str] = set()
            for e in direct_matches + ppr_edges:
                if e.edge_id not in seen_eids:
                    seen_eids.add(e.edge_id)
                    active_edges.append(e)
            active_edges = active_edges[:top_k_graph_edges]
            for e in direct_matches:
                reinforced_ids.add(e.edge_id)

        return intent, speaker_name, active_edges, historical_edges, multi_hop_paths, reinforced_ids

    def retrieve_sync(
        self,
        user_id: str,
        query: str,
        current_turn: int,
        top_k_episodes: int = 3,
        top_k_graph_edges: int = 10,
        reinforce_on_read: bool = True,
        at_turn: int | None = None,
    ) -> RetrievalResult:
        """Synchronously execute hybrid temporal-aware retrieval across Tier 1, 2, and 3."""
        with self._lock:
            if not self._initialized:
                self.initialize_sync()

            t0 = time.perf_counter()
            (
                intent,
                speaker_name,
                active_edges,
                historical_edges,
                multi_hop_paths,
                reinforced_ids,
            ) = self._compute_graph_retrieval(
                user_id, query, current_turn, top_k_graph_edges, at_turn
            )

            suppressed_count = 0
            if at_turn is not None:
                episodic_hits = []
            elif intent == QueryIntent.HISTORICAL_LINEAGE:
                episodic_hits = self.episodic_store.search_sync(
                    query=query, user_id=user_id, top_k=top_k_episodes
                )
            else:
                superseded_eps = self.temporal_graph.get_superseded_episode_ids(user_id)
                raw_hits = self.episodic_store.search_sync(
                    query=query, user_id=user_id, top_k=top_k_episodes + 4
                )
                episodic_hits = []
                for ch, sim in raw_hits:
                    if ch.episode_id in superseded_eps:
                        suppressed_count += 1
                        continue
                    episodic_hits.append((ch, sim))
                    if len(episodic_hits) >= top_k_episodes:
                        break

            if reinforce_on_read and reinforced_ids:
                self.temporal_graph.reinforce_edges_sync(reinforced_ids, current_turn=current_turn)

            superseded_eps_all = (
                self.temporal_graph.get_superseded_episode_ids(user_id)
                if intent == QueryIntent.CURRENT_STATE
                else set()
            )
            working_turns = [
                wt
                for wt in self.working_memory.get_recent_turns(user_id, limit=3)
                if f"ep_{wt.user_id}_{wt.turn_id:04d}" not in superseded_eps_all
                and wt.salience_score >= self.working_memory.salience_threshold
            ]

            formatted = self._format_prompt_context(
                intent=intent,
                active_edges=active_edges,
                historical_edges=historical_edges,
                multi_hop_paths=multi_hop_paths,
                episodic_hits=episodic_hits,
                working_turns=working_turns,
                speaker_name=speaker_name,
            )
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

            return RetrievalResult(
                query=query,
                user_id=user_id,
                intent=intent,
                current_turn=current_turn,
                working_turns=working_turns,
                active_edges=active_edges,
                historical_edges=historical_edges,
                multi_hop_paths=multi_hop_paths,
                episodic_hits=episodic_hits,
                suppressed_conflicting_episodes=suppressed_count,
                formatted_prompt_context=formatted,
                prompt_tokens=estimate_tokens(formatted),
                latency_ms=latency_ms,
            )

    async def retrieve(
        self,
        user_id: str,
        query: str,
        current_turn: int,
        top_k_episodes: int = 3,
        top_k_graph_edges: int = 10,
        reinforce_on_read: bool = True,
        at_turn: int | None = None,
    ) -> RetrievalResult:
        """Asynchronously execute hybrid temporal-aware retrieval across Tier 1, 2, and 3."""
        if not self._initialized:
            await self.initialize()

        t0 = time.perf_counter()
        (
            intent,
            speaker_name,
            active_edges,
            historical_edges,
            multi_hop_paths,
            reinforced_ids,
        ) = self._compute_graph_retrieval(
            user_id, query, current_turn, top_k_graph_edges, at_turn
        )

        suppressed_count = 0
        if at_turn is not None:
            episodic_hits = []
        elif intent == QueryIntent.HISTORICAL_LINEAGE:
            episodic_hits = await self.episodic_store.search(
                query=query, user_id=user_id, top_k=top_k_episodes
            )
        else:
            superseded_eps = self.temporal_graph.get_superseded_episode_ids(user_id)
            raw_hits = await self.episodic_store.search(
                query=query, user_id=user_id, top_k=top_k_episodes + 4
            )
            episodic_hits = []
            for ch, sim in raw_hits:
                if ch.episode_id in superseded_eps:
                    suppressed_count += 1
                    continue
                episodic_hits.append((ch, sim))
                if len(episodic_hits) >= top_k_episodes:
                    break

        if reinforce_on_read and reinforced_ids:
            await self.temporal_graph.reinforce_edges(reinforced_ids, current_turn=current_turn)

        superseded_eps_all = (
            self.temporal_graph.get_superseded_episode_ids(user_id)
            if intent == QueryIntent.CURRENT_STATE
            else set()
        )
        working_turns = [
            wt
            for wt in self.working_memory.get_recent_turns(user_id, limit=3)
            if f"ep_{wt.user_id}_{wt.turn_id:04d}" not in superseded_eps_all
            and wt.salience_score >= self.working_memory.salience_threshold
        ]

        formatted = self._format_prompt_context(
            intent=intent,
            active_edges=active_edges,
            historical_edges=historical_edges,
            multi_hop_paths=multi_hop_paths,
            episodic_hits=episodic_hits,
            working_turns=working_turns,
            speaker_name=speaker_name,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        return RetrievalResult(
            query=query,
            user_id=user_id,
            intent=intent,
            current_turn=current_turn,
            working_turns=working_turns,
            active_edges=active_edges,
            historical_edges=historical_edges,
            multi_hop_paths=multi_hop_paths,
            episodic_hits=episodic_hits,
            suppressed_conflicting_episodes=suppressed_count,
            formatted_prompt_context=formatted,
            prompt_tokens=estimate_tokens(formatted),
            latency_ms=latency_ms,
        )

    def answer_query_sync(
        self,
        user_id: str,
        query: str,
        current_turn: int,
        reinforce_on_read: bool = True,
    ) -> tuple[str, RetrievalResult]:
        res = self.retrieve_sync(
            user_id=user_id,
            query=query,
            current_turn=current_turn,
            reinforce_on_read=reinforce_on_read,
        )
        ans = self.reader.answer_offline(
            question=query,
            prompt_context=res.formatted_prompt_context,
            user_display_name=self.get_user_display_name(user_id),
        )
        return ans, res

    async def answer_query(
        self,
        user_id: str,
        query: str,
        current_turn: int,
        reinforce_on_read: bool = True,
    ) -> tuple[str, RetrievalResult]:
        res = await self.retrieve(
            user_id=user_id,
            query=query,
            current_turn=current_turn,
            reinforce_on_read=reinforce_on_read,
        )
        ans = await self.reader.aanswer(
            question=query,
            prompt_context=res.formatted_prompt_context,
            user_display_name=self.get_user_display_name(user_id),
        )
        return ans, res

    def _format_prompt_context(
        self,
        intent: QueryIntent,
        active_edges: list[TemporalEdge],
        historical_edges: list[TemporalEdge],
        multi_hop_paths: list[list[TemporalEdge]],
        episodic_hits: list[tuple],
        working_turns: list[DialogueTurn],
        speaker_name: str,
    ) -> str:
        lines: list[str] = []

        if intent == QueryIntent.HISTORICAL_LINEAGE and historical_edges:
            grouped: dict[tuple[str, str], list[TemporalEdge]] = {}
            for e in historical_edges:
                grouped.setdefault((e.subject, e.predicate), []).append(e)
            for (subj, pred), chain in grouped.items():
                chain_sorted = sorted(chain, key=lambda x: x.valid_from_turn)
                s_disp = self.temporal_graph.display_name(subj)
                parts = []
                for step in chain_sorted:
                    o_disp = self.temporal_graph.display_name(step.object)
                    until_s = (
                        f"Turn {step.valid_until_turn}"
                        if step.valid_until_turn is not None
                        else "Present"
                    )
                    st = "ACTIVE" if step.valid_until_turn is None else "SUPERSEDED"
                    parts.append(f"{o_disp} [Turn {step.valid_from_turn} -> {until_s}, {st}]")
                lines.append(f"[LINEAGE] ({s_disp}, {pred}): " + " ==> ".join(parts))

        if multi_hop_paths:
            for path in multi_hop_paths[:4]:
                hop_strs = [
                    self.temporal_graph.format_edge_readable(e, include_validity=False)
                    for e in path
                ]
                lines.append("[MULTI-HOP PATH] " + " -> ".join(hop_strs))

        for edge in active_edges:
            lines.append(
                f"[ACTIVE FACT] {self.temporal_graph.format_edge_readable(edge, include_validity=False)}"
            )

        seen_turns: set[int] = set()
        for chunk, sim in episodic_hits[:2]:
            seen_turns.add(chunk.turn_id)
            lines.append(f"[Turn {chunk.turn_id} | sim={sim:.2f}] {speaker_name}: {chunk.content}")

        for wt in working_turns:
            if wt.turn_id not in seen_turns:
                lines.append(f"[Turn {wt.turn_id} | working] {speaker_name}: {wt.content}")

        return "\n".join(lines)

    def close_sync(self) -> None:
        if self._thread_worker is not None and self._thread_worker.is_alive():
            self._thread_queue.put(None)
            self._thread_worker.join(timeout=2.0)
        self.episodic_store.close_sync()
        self.temporal_graph.close_sync()

    async def close(self) -> None:
        if self._bg_worker_task is not None and not self._bg_worker_task.done():
            assert self._bg_queue is not None
            await self._bg_queue.put(None)
            await self._bg_worker_task
        self.close_sync()
