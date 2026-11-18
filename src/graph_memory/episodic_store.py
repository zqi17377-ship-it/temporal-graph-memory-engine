"""Tier 2 Episodic Vector Store: SQLite-backed episodic chunks with fast dense cosine search."""

from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Iterable
import uuid

import aiosqlite
import numpy as np

from graph_memory.embeddings import EmbeddingService
from graph_memory.models import DialogueTurn, EpisodicChunk

_CREATE_EPISODES_SQL = """
CREATE TABLE IF NOT EXISTS episodes (
    episode_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    turn_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    content TEXT NOT NULL,
    salience_score REAL NOT NULL,
    token_count INTEGER NOT NULL,
    embedding_blob BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episodes_user_turn ON episodes(user_id, turn_id);
"""


class EpisodicVectorStore:
    """SQLite-persisted episodic memory store with vectorized NumPy cosine similarity search."""

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        embedder: EmbeddingService | None = None,
    ) -> None:
        self.db_path = str(db_path)
        self.embedder = embedder or EmbeddingService()
        self._is_memory = self.db_path == ":memory:"
        self._mem_conn: sqlite3.Connection | None = (
            sqlite3.connect(":memory:", check_same_thread=False) if self._is_memory else None
        )
        self._chunks: list[EpisodicChunk] = []
        self._id_to_idx: dict[str, int] = {}
        self._matrix: np.ndarray = np.zeros((0, self.embedder.dim), dtype=np.float32)
        self._initialized = False

    def _hydrate_rows(self, rows: list[tuple]) -> None:
        self._chunks.clear()
        self._id_to_idx.clear()
        vecs: list[np.ndarray] = []

        for row in rows:
            vec = np.frombuffer(row[8], dtype=np.float32).copy()
            chunk = EpisodicChunk(
                episode_id=row[0],
                user_id=row[1],
                session_id=row[2],
                turn_id=row[3],
                timestamp=row[4],
                content=row[5],
                salience_score=float(row[6]),
                token_count=int(row[7]),
                embedding=vec.tolist(),
            )
            self._id_to_idx[chunk.episode_id] = len(self._chunks)
            self._chunks.append(chunk)
            vecs.append(vec)

        if vecs:
            self._matrix = np.vstack(vecs).astype(np.float32)
        else:
            self._matrix = np.zeros((0, self.embedder.dim), dtype=np.float32)
        self._initialized = True

    def initialize_sync(self) -> None:
        """Synchronously create SQLite schema and hydrate normalized dense vector index."""
        query_sql = (
            "SELECT episode_id, user_id, session_id, turn_id, timestamp, content, "
            "salience_score, token_count, embedding_blob "
            "FROM episodes ORDER BY turn_id ASC, rowid ASC"
        )
        if self._is_memory:
            assert self._mem_conn is not None
            self._mem_conn.executescript(_CREATE_EPISODES_SQL)
            self._mem_conn.commit()
            rows = self._mem_conn.execute(query_sql).fetchall()
        else:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                conn.executescript(_CREATE_EPISODES_SQL)
                conn.commit()
                rows = conn.execute(query_sql).fetchall()
        self._hydrate_rows(rows)

    async def initialize(self) -> None:
        """Asynchronously create SQLite schema via aiosqlite and hydrate vector matrix."""
        if self._is_memory:
            self.initialize_sync()
            return
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(_CREATE_EPISODES_SQL)
            await db.commit()
            async with db.execute(
                "SELECT episode_id, user_id, session_id, turn_id, timestamp, content, "
                "salience_score, token_count, embedding_blob "
                "FROM episodes ORDER BY turn_id ASC, rowid ASC"
            ) as cursor:
                rows = await cursor.fetchall()
        self._hydrate_rows(list(rows))

    def _append_chunk_in_memory(self, chunk: EpisodicChunk, vec: np.ndarray) -> None:
        self._id_to_idx[chunk.episode_id] = len(self._chunks)
        self._chunks.append(chunk)
        if self._matrix.shape[0] == 0:
            self._matrix = vec.reshape(1, -1).astype(np.float32)
        else:
            self._matrix = np.vstack([self._matrix, vec.reshape(1, -1)]).astype(np.float32)

    def add_turn_as_episode_sync(
        self,
        turn: DialogueTurn,
        episode_id: str | None = None,
    ) -> EpisodicChunk:
        """Synchronously embed a salient DialogueTurn, persist to SQLite, and index vector."""
        if not self._initialized:
            self.initialize_sync()

        ep_id = episode_id or f"ep_{turn.user_id}_{turn.turn_id:04d}_{uuid.uuid4().hex[:6]}"
        if ep_id in self._id_to_idx:
            return self._chunks[self._id_to_idx[ep_id]]

        vec = self.embedder.embed_text(turn.content)
        blob = vec.astype(np.float32).tobytes()
        chunk = EpisodicChunk(
            episode_id=ep_id,
            user_id=turn.user_id,
            session_id=turn.session_id,
            turn_id=turn.turn_id,
            timestamp=turn.timestamp,
            content=turn.content,
            salience_score=turn.salience_score,
            token_count=turn.estimated_tokens,
            embedding=vec.tolist(),
        )
        params = (
            chunk.episode_id,
            chunk.user_id,
            chunk.session_id,
            chunk.turn_id,
            chunk.timestamp,
            chunk.content,
            chunk.salience_score,
            chunk.token_count,
            blob,
        )
        if self._is_memory:
            assert self._mem_conn is not None
            self._mem_conn.execute(
                "INSERT OR REPLACE INTO episodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", params
            )
            self._mem_conn.commit()
        else:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO episodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", params
                )
                conn.commit()

        self._append_chunk_in_memory(chunk, vec)
        return chunk

    async def add_turn_as_episode(
        self,
        turn: DialogueTurn,
        episode_id: str | None = None,
    ) -> EpisodicChunk:
        """Asynchronously embed a salient DialogueTurn and persist via aiosqlite."""
        if not self._initialized:
            await self.initialize()

        ep_id = episode_id or f"ep_{turn.user_id}_{turn.turn_id:04d}_{uuid.uuid4().hex[:6]}"
        if ep_id in self._id_to_idx:
            return self._chunks[self._id_to_idx[ep_id]]

        vec = await self.embedder.aembed_text(turn.content)
        blob = vec.astype(np.float32).tobytes()
        chunk = EpisodicChunk(
            episode_id=ep_id,
            user_id=turn.user_id,
            session_id=turn.session_id,
            turn_id=turn.turn_id,
            timestamp=turn.timestamp,
            content=turn.content,
            salience_score=turn.salience_score,
            token_count=turn.estimated_tokens,
            embedding=vec.tolist(),
        )
        params = (
            chunk.episode_id,
            chunk.user_id,
            chunk.session_id,
            chunk.turn_id,
            chunk.timestamp,
            chunk.content,
            chunk.salience_score,
            chunk.token_count,
            blob,
        )
        if self._is_memory:
            assert self._mem_conn is not None
            self._mem_conn.execute(
                "INSERT OR REPLACE INTO episodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", params
            )
            self._mem_conn.commit()
        else:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "INSERT OR REPLACE INTO episodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", params
                )
                await db.commit()

        self._append_chunk_in_memory(chunk, vec)
        return chunk

    def _rank_by_vector(
        self,
        q_vec: np.ndarray,
        user_id: str | None = None,
        top_k: int = 5,
        min_similarity: float = 0.02,
        exclude_episode_ids: Iterable[str] | None = None,
    ) -> list[tuple[EpisodicChunk, float]]:
        if not self._chunks or self._matrix.shape[0] == 0:
            return []
        raw_scores = self._matrix @ q_vec
        excluded = set(exclude_episode_ids) if exclude_episode_ids else set()
        candidates: list[tuple[int, float]] = []
        for idx, chunk in enumerate(self._chunks):
            if user_id is not None and chunk.user_id != user_id:
                continue
            if chunk.episode_id in excluded:
                continue
            sim = float(raw_scores[idx])
            if sim >= min_similarity:
                candidates.append((idx, sim))
        candidates.sort(key=lambda item: item[1], reverse=True)
        return [(self._chunks[idx], round(sim, 4)) for idx, sim in candidates[:top_k]]

    def search_sync(
        self,
        query: str,
        user_id: str | None = None,
        top_k: int = 5,
        min_similarity: float = 0.02,
        exclude_episode_ids: Iterable[str] | None = None,
    ) -> list[tuple[EpisodicChunk, float]]:
        """Synchronously search stored episodes by cosine similarity."""
        if not self._initialized:
            self.initialize_sync()
        if not self._chunks:
            return []
        q_vec = self.embedder.embed_text(query)
        return self._rank_by_vector(q_vec, user_id, top_k, min_similarity, exclude_episode_ids)

    async def search(
        self,
        query: str,
        user_id: str | None = None,
        top_k: int = 5,
        min_similarity: float = 0.02,
        exclude_episode_ids: Iterable[str] | None = None,
    ) -> list[tuple[EpisodicChunk, float]]:
        """Asynchronously search stored episodes by cosine similarity."""
        if not self._initialized:
            await self.initialize()
        if not self._chunks:
            return []
        q_vec = await self.embedder.aembed_text(query)
        return self._rank_by_vector(q_vec, user_id, top_k, min_similarity, exclude_episode_ids)

    def count(self, user_id: str | None = None) -> int:
        if user_id is None:
            return len(self._chunks)
        return sum(1 for c in self._chunks if c.user_id == user_id)

    def get_all_episodes(self, user_id: str | None = None) -> list[EpisodicChunk]:
        if user_id is None:
            return list(self._chunks)
        return [c for c in self._chunks if c.user_id == user_id]

    def close_sync(self) -> None:
        if self._mem_conn is not None:
            self._mem_conn.close()
            self._mem_conn = None

    async def close(self) -> None:
        self.close_sync()
