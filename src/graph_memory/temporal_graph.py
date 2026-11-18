"""Tier 3 Temporal Knowledge Graph: SQLite persistence, alias canonicalization, and PPR/BFS."""

from __future__ import annotations

from collections import defaultdict, deque
import json
from pathlib import Path
import re
import sqlite3
from typing import Iterable, Sequence

import aiosqlite

from graph_memory.forgetting import ForgettingEngine
from graph_memory.models import EntityNode, EntityType, TemporalEdge

_CREATE_GRAPH_SQL = """
CREATE TABLE IF NOT EXISTS nodes (
    node_id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    aliases_json TEXT NOT NULL,
    created_at_turn INTEGER NOT NULL,
    last_seen_turn INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS temporal_edges (
    edge_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    valid_from_turn INTEGER NOT NULL,
    valid_until_turn INTEGER,
    confidence REAL NOT NULL,
    base_importance REAL NOT NULL,
    access_count INTEGER NOT NULL,
    last_accessed_turn INTEGER NOT NULL,
    source_episode_id TEXT NOT NULL,
    superseded_by TEXT,
    archived INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_edges_user_subj_pred
    ON temporal_edges(user_id, subject, predicate);
CREATE INDEX IF NOT EXISTS idx_edges_validity
    ON temporal_edges(user_id, valid_from_turn, valid_until_turn);
"""

_BUILTIN_ALIASES: dict[str, tuple[str, EntityType]] = {
    "postgres": ("PostgreSQL", EntityType.TECH_STACK),
    "psql": ("PostgreSQL", EntityType.TECH_STACK),
    "postgresql": ("PostgreSQL", EntityType.TECH_STACK),
    "postgresql 16": ("PostgreSQL", EntityType.TECH_STACK),
    "mysql": ("MySQL", EntityType.TECH_STACK),
    "mysql 8": ("MySQL", EntityType.TECH_STACK),
    "mongo": ("MongoDB", EntityType.TECH_STACK),
    "mongodb": ("MongoDB", EntityType.TECH_STACK),
    "sqlite": ("SQLite", EntityType.TECH_STACK),
    "sqlite3": ("SQLite", EntityType.TECH_STACK),
    "redis": ("Redis", EntityType.TECH_STACK),
    "clickhouse": ("ClickHouse", EntityType.TECH_STACK),
    "duckdb": ("DuckDB", EntityType.TECH_STACK),
    "milvus": ("Milvus", EntityType.TECH_STACK),
    "qdrant": ("Qdrant", EntityType.TECH_STACK),
    "neo4j": ("Neo4j", EntityType.TECH_STACK),
    "torch": ("PyTorch", EntityType.TECH_STACK),
    "pytorch": ("PyTorch", EntityType.TECH_STACK),
    "tf": ("TensorFlow", EntityType.TECH_STACK),
    "tensorflow": ("TensorFlow", EntityType.TECH_STACK),
    "jax": ("JAX", EntityType.TECH_STACK),
    "triton": ("Triton", EntityType.TECH_STACK),
    "vllm": ("vLLM", EntityType.TECH_STACK),
    "langchain": ("LangChain", EntityType.TECH_STACK),
    "llamaindex": ("LlamaIndex", EntityType.TECH_STACK),
    "fastapi": ("FastAPI", EntityType.TECH_STACK),
    "ray": ("Ray", EntityType.TECH_STACK),
    "k8s": ("Kubernetes", EntityType.TECH_STACK),
    "kubernetes": ("Kubernetes", EntityType.TECH_STACK),
    "bj": ("Beijing", EntityType.CITY),
    "peking": ("Beijing", EntityType.CITY),
    "beijing": ("Beijing", EntityType.CITY),
    "hz": ("Hangzhou", EntityType.CITY),
    "hangzhou": ("Hangzhou", EntityType.CITY),
    "sh": ("Shanghai", EntityType.CITY),
    "shanghai": ("Shanghai", EntityType.CITY),
    "sz": ("Shenzhen", EntityType.CITY),
    "shenzhen": ("Shenzhen", EntityType.CITY),
    "gz": ("Guangzhou", EntityType.CITY),
    "guangzhou": ("Guangzhou", EntityType.CITY),
    "nanjing": ("Nanjing", EntityType.CITY),
    "chengdu": ("Chengdu", EntityType.CITY),
    "tokyo": ("Tokyo", EntityType.CITY),
    "singapore": ("Singapore", EntityType.CITY),
    "neurips": ("NeurIPS", EntityType.ORGANIZATION),
    "nips": ("NeurIPS", EntityType.ORGANIZATION),
    "icml": ("ICML", EntityType.ORGANIZATION),
    "iclr": ("ICLR", EntityType.ORGANIZATION),
    "cvpr": ("CVPR", EntityType.ORGANIZATION),
    "acl": ("ACL", EntityType.ORGANIZATION),
    "emnlp": ("EMNLP", EntityType.ORGANIZATION),
    "osdi": ("OSDI", EntityType.ORGANIZATION),
    "sigmod": ("SIGMOD", EntityType.ORGANIZATION),
    "vldb": ("VLDB", EntityType.ORGANIZATION),
    "tsinghua air": ("Tsinghua AIR", EntityType.ORGANIZATION),
    "shanghai ai lab": ("Shanghai AI Lab", EntityType.ORGANIZATION),
    "shailab": ("Shanghai AI Lab", EntityType.ORGANIZATION),
    "zhejiang lab": ("Zhejiang Lab", EntityType.ORGANIZATION),
    "zjlab": ("Zhejiang Lab", EntityType.ORGANIZATION),
    "msra": ("MSRA", EntityType.ORGANIZATION),
    "deepseek": ("DeepSeek", EntityType.ORGANIZATION),
    "bytedance": ("ByteDance", EntityType.ORGANIZATION),
    "alibaba": ("Alibaba Cloud", EntityType.ORGANIZATION),
    "alibaba cloud": ("Alibaba Cloud", EntityType.ORGANIZATION),
    "tencent": ("Tencent AI Lab", EntityType.ORGANIZATION),
    "tencent ai lab": ("Tencent AI Lab", EntityType.ORGANIZATION),
}


def _normalize_key(raw: str) -> str:
    cleaned = re.sub(r"\s+", " ", raw.strip().lower())
    cleaned = re.sub(r"^the\s+", "", cleaned)
    return cleaned


def _slugify(raw: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", _normalize_key(raw)).strip("_")
    return slug or "entity"


class TemporalKnowledgeGraph:
    """SQLite-persisted Temporal Property Graph with alias canonicalization and PPR/BFS."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._is_memory = self.db_path == ":memory:"
        self._mem_conn: sqlite3.Connection | None = (
            sqlite3.connect(":memory:", check_same_thread=False) if self._is_memory else None
        )
        self._nodes: dict[str, EntityNode] = {}
        self._alias_to_node_id: dict[str, str] = {}
        self._custom_aliases: dict[str, tuple[str, EntityType]] = dict(_BUILTIN_ALIASES)
        self._edges: dict[str, TemporalEdge] = {}
        self._initialized = False

    def _hydrate_graph(self, node_rows: list[tuple], edge_rows: list[tuple]) -> None:
        self._nodes.clear()
        self._alias_to_node_id.clear()
        self._edges.clear()

        for row in node_rows:
            aliases = json.loads(row[3])
            node = EntityNode(
                node_id=row[0],
                canonical_name=row[1],
                entity_type=EntityType(row[2]),
                aliases=aliases,
                created_at_turn=int(row[4]),
                last_seen_turn=int(row[5]),
            )
            self._nodes[node.node_id] = node
            self._alias_to_node_id[_normalize_key(node.canonical_name)] = node.node_id
            self._alias_to_node_id[node.node_id] = node.node_id
            for al in aliases:
                self._alias_to_node_id[_normalize_key(al)] = node.node_id

        for row in edge_rows:
            edge = TemporalEdge(
                edge_id=row[0],
                user_id=row[1],
                subject=row[2],
                predicate=row[3],
                object=row[4],
                valid_from_turn=int(row[5]),
                valid_until_turn=int(row[6]) if row[6] is not None else None,
                confidence=float(row[7]),
                base_importance=float(row[8]),
                access_count=int(row[9]),
                last_accessed_turn=int(row[10]),
                source_episode_id=row[11],
                superseded_by=row[12],
                archived=bool(row[13]),
            )
            self._edges[edge.edge_id] = edge

        self._initialized = True

    def initialize_sync(self) -> None:
        """Synchronously initialize SQLite schema and load persisted graph into memory."""
        n_sql = (
            "SELECT node_id, canonical_name, entity_type, aliases_json, "
            "created_at_turn, last_seen_turn FROM nodes"
        )
        e_sql = (
            "SELECT edge_id, user_id, subject, predicate, object, valid_from_turn, "
            "valid_until_turn, confidence, base_importance, access_count, "
            "last_accessed_turn, source_episode_id, superseded_by, archived "
            "FROM temporal_edges ORDER BY valid_from_turn ASC, rowid ASC"
        )
        if self._is_memory:
            assert self._mem_conn is not None
            self._mem_conn.executescript(_CREATE_GRAPH_SQL)
            self._mem_conn.commit()
            node_rows = self._mem_conn.execute(n_sql).fetchall()
            edge_rows = self._mem_conn.execute(e_sql).fetchall()
        else:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                conn.executescript(_CREATE_GRAPH_SQL)
                conn.commit()
                node_rows = conn.execute(n_sql).fetchall()
                edge_rows = conn.execute(e_sql).fetchall()
        self._hydrate_graph(node_rows, edge_rows)

    async def initialize(self) -> None:
        """Asynchronously initialize SQLite schema via aiosqlite and load persisted graph."""
        if self._is_memory:
            self.initialize_sync()
            return
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(_CREATE_GRAPH_SQL)
            await db.commit()
            async with db.execute(
                "SELECT node_id, canonical_name, entity_type, aliases_json, "
                "created_at_turn, last_seen_turn FROM nodes"
            ) as cur:
                node_rows = await cur.fetchall()
            async with db.execute(
                "SELECT edge_id, user_id, subject, predicate, object, valid_from_turn, "
                "valid_until_turn, confidence, base_importance, access_count, "
                "last_accessed_turn, source_episode_id, superseded_by, archived "
                "FROM temporal_edges ORDER BY valid_from_turn ASC, rowid ASC"
            ) as cur:
                edge_rows = await cur.fetchall()
        self._hydrate_graph(list(node_rows), list(edge_rows))

    def register_alias(
        self,
        alias: str,
        canonical_name: str,
        entity_type: EntityType,
    ) -> None:
        """Register an entity alias mapping to a canonical name and EntityType."""
        norm_alias = _normalize_key(alias)
        self._custom_aliases[norm_alias] = (canonical_name, entity_type)
        norm_canon = _normalize_key(canonical_name)
        if norm_canon in self._alias_to_node_id:
            nid = self._alias_to_node_id[norm_canon]
            self._alias_to_node_id[norm_alias] = nid
            node = self._nodes[nid]
            if alias not in node.aliases:
                node.aliases.append(alias)

    def resolve_entity_id(self, raw_name: str) -> str | None:
        """Resolve a raw name or alias to an existing node_id if known."""
        key = _normalize_key(raw_name)
        if key in self._alias_to_node_id:
            return self._alias_to_node_id[key]
        if key in self._custom_aliases:
            canon_name, _ = self._custom_aliases[key]
            return self._alias_to_node_id.get(_normalize_key(canon_name))
        return None

    def get_node(self, node_id: str) -> EntityNode | None:
        return self._nodes.get(node_id)

    def display_name(self, node_id_or_name: str) -> str:
        node = self._nodes.get(node_id_or_name)
        return node.canonical_name if node else node_id_or_name

    def _prepare_node_upsert(
        self,
        raw_name: str,
        entity_type: EntityType,
        current_turn: int,
        extra_aliases: Sequence[str] | None = None,
    ) -> tuple[EntityNode, tuple]:
        norm = _normalize_key(raw_name)
        canonical_name = raw_name.strip()
        resolved_type = entity_type

        if norm in self._custom_aliases:
            canonical_name, resolved_type = self._custom_aliases[norm]

        existing_id = self._alias_to_node_id.get(norm) or self._alias_to_node_id.get(
            _normalize_key(canonical_name)
        )

        if existing_id and existing_id in self._nodes:
            node = self._nodes[existing_id]
            new_aliases = list(node.aliases)
            if _normalize_key(raw_name) != _normalize_key(node.canonical_name):
                if raw_name.strip() not in new_aliases:
                    new_aliases.append(raw_name.strip())
            if extra_aliases:
                for al in extra_aliases:
                    if al.strip() and al.strip() not in new_aliases:
                        new_aliases.append(al.strip())
            node = node.model_copy(
                update={
                    "aliases": new_aliases,
                    "last_seen_turn": max(node.last_seen_turn, current_turn),
                }
            )
        else:
            node_id = f"{resolved_type.value.lower()}:{_slugify(canonical_name)}"
            aliases_list: list[str] = []
            if _normalize_key(raw_name) != _normalize_key(canonical_name):
                aliases_list.append(raw_name.strip())
            if extra_aliases:
                for al in extra_aliases:
                    if al.strip() and al.strip() not in aliases_list:
                        aliases_list.append(al.strip())
            node = EntityNode(
                node_id=node_id,
                canonical_name=canonical_name,
                entity_type=resolved_type,
                aliases=aliases_list,
                created_at_turn=current_turn,
                last_seen_turn=current_turn,
            )

        self._nodes[node.node_id] = node
        self._alias_to_node_id[_normalize_key(node.canonical_name)] = node.node_id
        self._alias_to_node_id[norm] = node.node_id
        self._alias_to_node_id[node.node_id] = node.node_id
        for al in node.aliases:
            self._alias_to_node_id[_normalize_key(al)] = node.node_id

        params = (
            node.node_id,
            node.canonical_name,
            node.entity_type.value,
            json.dumps(node.aliases, ensure_ascii=False),
            node.created_at_turn,
            node.last_seen_turn,
        )
        return node, params

    def upsert_node_sync(
        self,
        raw_name: str,
        entity_type: EntityType,
        current_turn: int,
        extra_aliases: Sequence[str] | None = None,
    ) -> EntityNode:
        if not self._initialized:
            self.initialize_sync()
        node, params = self._prepare_node_upsert(raw_name, entity_type, current_turn, extra_aliases)
        if self._is_memory:
            assert self._mem_conn is not None
            self._mem_conn.execute("INSERT OR REPLACE INTO nodes VALUES (?, ?, ?, ?, ?, ?)", params)
            self._mem_conn.commit()
        else:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("INSERT OR REPLACE INTO nodes VALUES (?, ?, ?, ?, ?, ?)", params)
                conn.commit()
        return node

    async def upsert_node(
        self,
        raw_name: str,
        entity_type: EntityType,
        current_turn: int,
        extra_aliases: Sequence[str] | None = None,
    ) -> EntityNode:
        if not self._initialized:
            await self.initialize()
        node, params = self._prepare_node_upsert(raw_name, entity_type, current_turn, extra_aliases)
        if self._is_memory:
            assert self._mem_conn is not None
            self._mem_conn.execute("INSERT OR REPLACE INTO nodes VALUES (?, ?, ?, ?, ?, ?)", params)
            self._mem_conn.commit()
        else:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("INSERT OR REPLACE INTO nodes VALUES (?, ?, ?, ?, ?, ?)", params)
                await db.commit()
        return node

    @staticmethod
    def _edge_params(edge: TemporalEdge) -> tuple:
        return (
            edge.edge_id,
            edge.user_id,
            edge.subject,
            edge.predicate,
            edge.object,
            edge.valid_from_turn,
            edge.valid_until_turn,
            edge.confidence,
            edge.base_importance,
            edge.access_count,
            edge.last_accessed_turn,
            edge.source_episode_id,
            edge.superseded_by,
            1 if edge.archived else 0,
        )

    def upsert_edge_sync(self, edge: TemporalEdge) -> TemporalEdge:
        if not self._initialized:
            self.initialize_sync()
        self._edges[edge.edge_id] = edge
        params = self._edge_params(edge)
        sql = "INSERT OR REPLACE INTO temporal_edges VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        if self._is_memory:
            assert self._mem_conn is not None
            self._mem_conn.execute(sql, params)
            self._mem_conn.commit()
        else:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(sql, params)
                conn.commit()
        return edge

    async def upsert_edge(self, edge: TemporalEdge) -> TemporalEdge:
        if not self._initialized:
            await self.initialize()
        self._edges[edge.edge_id] = edge
        params = self._edge_params(edge)
        sql = "INSERT OR REPLACE INTO temporal_edges VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        if self._is_memory:
            assert self._mem_conn is not None
            self._mem_conn.execute(sql, params)
            self._mem_conn.commit()
        else:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(sql, params)
                await db.commit()
        return edge

    def _prepare_reinforcement(
        self, edge_ids: Iterable[str], current_turn: int, boost: int = 1
    ) -> list[TemporalEdge]:
        updated: list[TemporalEdge] = []
        for eid in edge_ids:
            edge = self._edges.get(eid)
            if edge is None:
                continue
            new_edge = edge.model_copy(
                update={
                    "access_count": edge.access_count + boost,
                    "last_accessed_turn": max(edge.last_accessed_turn, current_turn),
                }
            )
            self._edges[eid] = new_edge
            updated.append(new_edge)
        return updated

    def reinforce_edges_sync(
        self, edge_ids: Iterable[str], current_turn: int, boost: int = 1
    ) -> None:
        updated = self._prepare_reinforcement(edge_ids, current_turn, boost)
        if not updated:
            return
        rows = [(e.access_count, e.last_accessed_turn, e.edge_id) for e in updated]
        sql = "UPDATE temporal_edges SET access_count = ?, last_accessed_turn = ? WHERE edge_id = ?"
        if self._is_memory:
            assert self._mem_conn is not None
            self._mem_conn.executemany(sql, rows)
            self._mem_conn.commit()
        else:
            with sqlite3.connect(self.db_path) as conn:
                conn.executemany(sql, rows)
                conn.commit()

    async def reinforce_edges(
        self, edge_ids: Iterable[str], current_turn: int, boost: int = 1
    ) -> None:
        updated = self._prepare_reinforcement(edge_ids, current_turn, boost)
        if not updated:
            return
        rows = [(e.access_count, e.last_accessed_turn, e.edge_id) for e in updated]
        sql = "UPDATE temporal_edges SET access_count = ?, last_accessed_turn = ? WHERE edge_id = ?"
        if self._is_memory:
            assert self._mem_conn is not None
            self._mem_conn.executemany(sql, rows)
            self._mem_conn.commit()
        else:
            async with aiosqlite.connect(self.db_path) as db:
                await db.executemany(sql, rows)
                await db.commit()

    def archive_edges_sync(self, edge_ids: Iterable[str]) -> None:
        ids = [eid for eid in edge_ids if eid in self._edges]
        if not ids:
            return
        for eid in ids:
            self._edges[eid] = self._edges[eid].model_copy(update={"archived": True})
        sql = "UPDATE temporal_edges SET archived = 1 WHERE edge_id = ?"
        if self._is_memory:
            assert self._mem_conn is not None
            self._mem_conn.executemany(sql, [(eid,) for eid in ids])
            self._mem_conn.commit()
        else:
            with sqlite3.connect(self.db_path) as conn:
                conn.executemany(sql, [(eid,) for eid in ids])
                conn.commit()

    async def archive_edges(self, edge_ids: Iterable[str]) -> None:
        ids = [eid for eid in edge_ids if eid in self._edges]
        if not ids:
            return
        for eid in ids:
            self._edges[eid] = self._edges[eid].model_copy(update={"archived": True})
        sql = "UPDATE temporal_edges SET archived = 1 WHERE edge_id = ?"
        if self._is_memory:
            assert self._mem_conn is not None
            self._mem_conn.executemany(sql, [(eid,) for eid in ids])
            self._mem_conn.commit()
        else:
            async with aiosqlite.connect(self.db_path) as db:
                await db.executemany(sql, [(eid,) for eid in ids])
                await db.commit()

    def get_all_nodes(self) -> list[EntityNode]:
        return list(self._nodes.values())

    def get_all_edges(self, user_id: str | None = None) -> list[TemporalEdge]:
        edges = list(self._edges.values())
        if user_id is not None:
            edges = [e for e in edges if e.user_id == user_id]
        return edges

    def get_active_edges(
        self,
        user_id: str | None = None,
        subject: str | None = None,
        predicate: str | None = None,
        object_id: str | None = None,
    ) -> list[TemporalEdge]:
        subj_resolved = self.resolve_entity_id(subject) if subject else None
        obj_resolved = self.resolve_entity_id(object_id) if object_id else None

        result: list[TemporalEdge] = []
        for edge in self._edges.values():
            if not edge.is_active:
                continue
            if user_id is not None and edge.user_id != user_id:
                continue
            if subject is not None and edge.subject not in (subject, subj_resolved):
                continue
            if predicate is not None and edge.predicate != predicate:
                continue
            if object_id is not None and edge.object not in (object_id, obj_resolved):
                continue
            result.append(edge)
        return result

    def get_edges_at_turn(
        self,
        turn_t: int,
        user_id: str | None = None,
        subject: str | None = None,
        predicate: str | None = None,
    ) -> list[TemporalEdge]:
        subj_resolved = self.resolve_entity_id(subject) if subject else None
        result: list[TemporalEdge] = []
        for edge in self._edges.values():
            if user_id is not None and edge.user_id != user_id:
                continue
            if subject is not None and edge.subject not in (subject, subj_resolved):
                continue
            if predicate is not None and edge.predicate != predicate:
                continue
            if edge.valid_from_turn <= turn_t and (
                edge.valid_until_turn is None or edge.valid_until_turn > turn_t
            ):
                result.append(edge)
        return sorted(result, key=lambda e: e.valid_from_turn)

    def get_predicate_lineage(
        self,
        subject: str,
        predicate: str | None = None,
        user_id: str | None = None,
    ) -> list[TemporalEdge]:
        subj_resolved = self.resolve_entity_id(subject) or subject
        lineage: list[TemporalEdge] = []
        for edge in self._edges.values():
            if user_id is not None and edge.user_id != user_id:
                continue
            if edge.subject != subj_resolved and edge.subject != subject:
                continue
            if predicate is not None and edge.predicate != predicate:
                continue
            lineage.append(edge)
        lineage.sort(key=lambda e: (e.valid_from_turn, 0 if e.valid_until_turn is not None else 1))
        return lineage

    def get_superseded_episode_ids(self, user_id: str) -> set[str]:
        active_ep_ids = {
            e.source_episode_id
            for e in self._edges.values()
            if e.user_id == user_id and e.is_active and e.source_episode_id
        }
        closed_ep_ids = {
            e.source_episode_id
            for e in self._edges.values()
            if e.user_id == user_id and e.valid_until_turn is not None and e.source_episode_id
        }
        return closed_ep_ids - active_ep_ids

    def match_entities_in_text(self, text: str) -> list[EntityNode]:
        norm_text = f" {_normalize_key(text)} "
        matched_ids: list[str] = []
        seen: set[str] = set()

        sorted_aliases = sorted(self._alias_to_node_id.items(), key=lambda kv: len(kv[0]), reverse=True)
        for alias_key, node_id in sorted_aliases:
            if ":" in alias_key or len(alias_key) < 2:
                continue
            if node_id in seen:
                continue
            pattern = rf"(?<![a-z0-9]){re.escape(alias_key)}(?![a-z0-9])"
            if re.search(pattern, norm_text):
                seen.add(node_id)
                matched_ids.append(node_id)

        return [self._nodes[nid] for nid in matched_ids if nid in self._nodes]

    def find_multi_hop_paths(
        self,
        seed_node_ids: Sequence[str],
        max_hops: int = 3,
        user_id: str | None = None,
        active_only: bool = True,
        at_turn: int | None = None,
        exclude_predicates: set[str] | None = None,
    ) -> list[list[TemporalEdge]]:
        if at_turn is not None:
            candidate_edges = self.get_edges_at_turn(at_turn, user_id=user_id)
        elif active_only:
            candidate_edges = self.get_active_edges(user_id=user_id)
        else:
            candidate_edges = [e for e in self.get_all_edges(user_id=user_id) if not e.archived]

        if exclude_predicates:
            candidate_edges = [e for e in candidate_edges if e.predicate not in exclude_predicates]

        adj: dict[str, list[tuple[str, TemporalEdge]]] = defaultdict(list)
        for edge in candidate_edges:
            adj[edge.subject].append((edge.object, edge))
            adj[edge.object].append((edge.subject, edge))

        resolved_seeds = [self.resolve_entity_id(s) or s for s in seed_node_ids]
        paths: list[list[TemporalEdge]] = []
        seen_edge_tuples: set[tuple[str, ...]] = set()

        for seed in resolved_seeds:
            if seed not in adj:
                continue
            queue: deque[tuple[str, list[TemporalEdge], set[str]]] = deque()
            queue.append((seed, [], {seed}))

            while queue:
                curr_node, curr_path, visited_nodes = queue.popleft()
                if len(curr_path) >= 2:
                    sig = tuple(e.edge_id for e in curr_path)
                    if sig not in seen_edge_tuples:
                        seen_edge_tuples.add(sig)
                        paths.append(curr_path)

                if len(curr_path) >= max_hops:
                    continue

                for neighbor, edge in adj[curr_node]:
                    if neighbor in visited_nodes:
                        continue
                    queue.append((neighbor, curr_path + [edge], visited_nodes | {neighbor}))

        paths.sort(
            key=lambda p: (
                sum(e.base_importance * e.confidence for e in p) / len(p),
                len(p),
            ),
            reverse=True,
        )
        return paths

    def personalized_pagerank(
        self,
        seed_node_ids: Sequence[str],
        current_turn: int,
        user_id: str | None = None,
        alpha: float = 0.85,
        max_iter: int = 30,
        top_k_edges: int = 12,
        active_only: bool = True,
        forgetting_engine: ForgettingEngine | None = None,
    ) -> list[TemporalEdge]:
        edges = (
            self.get_active_edges(user_id=user_id)
            if active_only
            else [e for e in self.get_all_edges(user_id=user_id) if not e.archived]
        )
        if not edges:
            return []

        node_set: set[str] = set()
        for e in edges:
            node_set.add(e.subject)
            node_set.add(e.object)

        resolved_seeds = [
            s_id
            for s in seed_node_ids
            if (s_id := (self.resolve_entity_id(s) or s)) in node_set
        ]
        if not resolved_seeds:
            resolved_seeds = list(node_set)

        nodes = sorted(node_set)
        idx_map = {n: i for i, n in enumerate(nodes)}
        n_nodes = len(nodes)

        teleport = [0.0] * n_nodes
        inv_seed = 1.0 / len(resolved_seeds)
        for s in resolved_seeds:
            teleport[idx_map[s]] += inv_seed

        edge_weights: dict[str, float] = {}
        neighbors: list[list[tuple[int, float]]] = [[] for _ in range(n_nodes)]
        out_weight = [0.0] * n_nodes

        for e in edges:
            ret = (
                forgetting_engine.retention_score(e, current_turn)
                if forgetting_engine is not None
                else e.base_importance
            )
            w = max(1e-4, e.confidence * ret)
            edge_weights[e.edge_id] = w
            u, v = idx_map[e.subject], idx_map[e.object]
            neighbors[u].append((v, w))
            neighbors[v].append((u, w))
            out_weight[u] += w
            out_weight[v] += w

        scores = list(teleport)
        for _ in range(max_iter):
            next_scores = [(1.0 - alpha) * t for t in teleport]
            dangling_mass = sum(scores[i] for i in range(n_nodes) if out_weight[i] <= 1e-12)
            if dangling_mass > 0.0:
                for i in range(n_nodes):
                    next_scores[i] += alpha * dangling_mass * teleport[i]

            for u in range(n_nodes):
                if out_weight[u] <= 1e-12:
                    continue
                share = alpha * scores[u] / out_weight[u]
                for v, w in neighbors[u]:
                    next_scores[v] += share * w

            diff = sum(abs(next_scores[i] - scores[i]) for i in range(n_nodes))
            scores = next_scores
            if diff < 1e-6:
                break

        scored_edges: list[tuple[float, TemporalEdge]] = []
        for e in edges:
            u_score = scores[idx_map[e.subject]]
            v_score = scores[idx_map[e.object]]
            rank_score = 0.5 * (u_score + v_score) * edge_weights[e.edge_id]
            scored_edges.append((rank_score, e))

        scored_edges.sort(key=lambda item: item[0], reverse=True)
        return [e for _, e in scored_edges[:top_k_edges]]

    def format_edge_readable(self, edge: TemporalEdge, include_validity: bool = False) -> str:
        s_name = self.display_name(edge.subject)
        o_name = self.display_name(edge.object)
        if not include_validity:
            return f"({s_name}, {edge.predicate}, {o_name})"
        until_str = f"Turn {edge.valid_until_turn}" if edge.valid_until_turn is not None else "Present"
        status = "ACTIVE" if edge.valid_until_turn is None else "SUPERSEDED"
        return (
            f"[{status} | Turn {edge.valid_from_turn} -> {until_str}] "
            f"({s_name}, {edge.predicate}, {o_name})"
        )

    def close_sync(self) -> None:
        if self._mem_conn is not None:
            self._mem_conn.close()
            self._mem_conn = None

    async def close(self) -> None:
        self.close_sync()
