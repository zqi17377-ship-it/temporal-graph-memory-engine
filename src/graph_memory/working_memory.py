"""Tier 1 Working Memory: Sliding short-term dialogue buffer with salience-gated eviction."""

from __future__ import annotations

from collections import deque
import re
from typing import Sequence

from graph_memory.models import DialogueTurn, ExtractedFact

_HIGH_SIGNAL_KEYWORDS = {
    "live", "moved", "relocated", "based", "work", "joined", "intern", "engineer",
    "role", "lead", "researcher", "database", "mysql", "postgresql", "postgres",
    "mongodb", "redis", "clickhouse", "sqlite", "project", "collaborat", "partner",
    "conference", "neurips", "icml", "iclr", "cvpr", "acl", "diet", "vegetarian",
    "vegan", "lactose", "halal", "gluten", "gpu", "rtx", "a100", "h100", "vram",
    "budget", "language", "fluent", "course", "library", "pytorch", "jax", "rust",
    "no longer", "stopped using", "deprecated", "switched", "migrated", "upgraded",
}

_LOW_SIGNAL_CHATTER_PATTERNS = re.compile(
    r"\b(weather|rain|coffee|latte|espresso|lunch|ramen|noodles|sleepy|tired|"
    r"haha|lol|weekend|movie|playlist|traffic|umbrella|keyboard switch|standup was boring)\b",
    re.IGNORECASE,
)


class WorkingMemory:
    """Sliding short-term dialogue buffer per user with salience-gated eviction to Tier 2."""

    def __init__(
        self,
        capacity: int = 6,
        salience_threshold: float = 0.35,
    ) -> None:
        if capacity < 1:
            raise ValueError("WorkingMemory capacity must be >= 1")
        self.capacity = capacity
        self.salience_threshold = salience_threshold
        self._buffers: dict[str, deque[DialogueTurn]] = {}
        self.total_ingested: int = 0
        self.evicted_promoted_count: int = 0
        self.evicted_discarded_count: int = 0

    def compute_salience(
        self,
        content: str,
        extracted_facts: Sequence[ExtractedFact] | None = None,
    ) -> float:
        """Estimate turn salience in [0.0, 1.0] from extracted triples and lexical density."""
        if extracted_facts:
            max_imp = max(f.base_importance * f.confidence for f in extracted_facts)
            # Core persona/project facts have base_importance >= 0.75 -> high salience
            # Transient ambient facts have base_importance <= 0.40 -> stays below or near gate
            bonus = min(0.12, 0.04 * (len(extracted_facts) - 1))
            return float(min(0.99, round(max_imp + bonus, 4)))

        lower = content.lower()
        words = content.split()
        score = 0.14

        matched_signals = sum(1 for kw in _HIGH_SIGNAL_KEYWORDS if kw in lower)
        score += min(0.45, matched_signals * 0.16)

        # Capitalized named entity heuristic (excluding sentence start)
        cap_tokens = sum(
            1 for tok in words[1:] if len(tok) > 1 and tok[0].isupper() and tok[0].isalpha()
        )
        score += min(0.22, cap_tokens * 0.08)

        # Numbers / specs (e.g., 24GB, 4090, CS224N)
        if re.search(r"\b\d+[A-Za-z]*\b", content):
            score += 0.08

        if _LOW_SIGNAL_CHATTER_PATTERNS.search(content) and matched_signals == 0:
            score *= 0.45

        return float(max(0.02, min(0.95, round(score, 4))))

    def push(
        self,
        turn: DialogueTurn,
        extracted_facts: Sequence[ExtractedFact] | None = None,
    ) -> DialogueTurn | None:
        """Insert a turn into the sliding buffer; return evicted turn ONLY if it passes salience gate."""
        self.total_ingested += 1
        if turn.salience_score <= 0.0:
            turn = turn.model_copy(
                update={"salience_score": self.compute_salience(turn.content, extracted_facts)}
            )

        buf = self._buffers.setdefault(turn.user_id, deque())
        evicted_for_promotion: DialogueTurn | None = None

        if len(buf) >= self.capacity:
            oldest = buf.popleft()
            if oldest.salience_score >= self.salience_threshold:
                self.evicted_promoted_count += 1
                evicted_for_promotion = oldest
            else:
                self.evicted_discarded_count += 1

        buf.append(turn)
        return evicted_for_promotion

    def flush_user(self, user_id: str) -> list[DialogueTurn]:
        """Evict all buffered turns for user_id and return those passing the salience gate."""
        buf = self._buffers.get(user_id)
        if not buf:
            return []
        promoted: list[DialogueTurn] = []
        while buf:
            oldest = buf.popleft()
            if oldest.salience_score >= self.salience_threshold:
                self.evicted_promoted_count += 1
                promoted.append(oldest)
            else:
                self.evicted_discarded_count += 1
        return promoted

    def get_recent_turns(self, user_id: str, limit: int | None = None) -> list[DialogueTurn]:
        """Return active short-term dialogue turns currently held in Working Memory."""
        buf = self._buffers.get(user_id)
        if not buf:
            return []
        items = list(buf)
        if limit is not None and limit > 0:
            return items[-limit:]
        return items

    def size(self, user_id: str) -> int:
        buf = self._buffers.get(user_id)
        return len(buf) if buf else 0
