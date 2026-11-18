"""Core Pydantic v2 data models for the Hierarchical & Temporal Graph Memory Engine."""

from __future__ import annotations

from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """Canonical entity types stored in Tier 3 Temporal Knowledge Graph."""

    PERSON = "Person"
    CITY = "City"
    PROJECT = "Project"
    TECH_STACK = "TechStack"
    PREFERENCE = "Preference"
    CONSTRAINT = "Constraint"
    ORGANIZATION = "Organization"


class PredicateCategory(str, Enum):
    """Semantic cardinality/update category for knowledge graph predicates."""

    FUNCTIONAL = "functional"
    ADDITIVE = "additive"
    RETRACTION = "retraction"


class QueryIntent(str, Enum):
    """Temporal and relational query intent detected by the hybrid retriever."""

    CURRENT_STATE = "current_state"
    HISTORICAL_LINEAGE = "historical_lineage"
    MULTI_HOP_RELATIONAL = "multi_hop_relational"


class DialogueTurn(BaseModel):
    """Single conversational turn entering Tier 1 Working Memory."""

    turn_id: int
    session_id: str
    user_id: str
    role: Literal["user", "assistant"] = "user"
    content: str
    timestamp: str = "2026-01-01T00:00:00Z"
    salience_score: float = 0.0

    @property
    def estimated_tokens(self) -> int:
        return max(1, int(len(self.content.split()) * 1.35))


class EpisodicChunk(BaseModel):
    """Salient conversational episode persisted in Tier 2 Episodic Vector Store."""

    episode_id: str
    user_id: str
    session_id: str
    turn_id: int
    timestamp: str
    content: str
    salience_score: float
    token_count: int
    embedding: list[float] = Field(default_factory=list)


class EntityNode(BaseModel):
    """Typed entity node with canonicalization and alias tracking in Tier 3."""

    node_id: str
    canonical_name: str
    entity_type: EntityType
    aliases: list[str] = Field(default_factory=list)
    created_at_turn: int
    last_seen_turn: int


class TemporalEdge(BaseModel):
    """Bi-temporal / validity-interval directed edge in Tier 3 Temporal Knowledge Graph."""

    edge_id: str
    user_id: str = "default_user"
    subject: str
    predicate: str
    object: str
    valid_from_turn: int
    valid_until_turn: int | None = None
    confidence: float = 0.90
    base_importance: float = 0.85
    access_count: int = 0
    last_accessed_turn: int = 0
    source_episode_id: str = ""
    superseded_by: str | None = None
    archived: bool = False

    @property
    def is_active(self) -> bool:
        return self.valid_until_turn is None and not self.archived


class ExtractedFact(BaseModel):
    """Structured candidate triple extracted from a dialogue turn."""

    subject: str
    subject_type: EntityType
    predicate: str
    object: str
    object_type: EntityType
    confidence: float = 0.92
    base_importance: float = 0.85
    is_retraction: bool = False
    target_predicate: str | None = None


class ConflictResolutionEvent(BaseModel):
    """Audit record produced when resolving functional overrides, additions, or retractions."""

    action: Literal["INSERTED", "SUPERSEDED", "ADDED_COEXISTING", "REINFORCED", "RETRACTED"]
    turn: int
    subject: str
    predicate: str
    new_object: str
    old_object: str | None = None
    new_edge_id: str | None = None
    closed_edge_ids: list[str] = Field(default_factory=list)


class ConsolidationReport(BaseModel):
    """Summary of an Ebbinghaus time-decay consolidation pass."""

    current_turn: int
    threshold_theta: float
    total_active_before: int
    archived_count: int
    retained_count: int
    compression_ratio: float
    archived_edge_ids: list[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    """Hybrid retrieval payload combining Tier 1, Tier 2, and Tier 3 context."""

    query: str
    user_id: str
    intent: QueryIntent
    current_turn: int
    working_turns: list[DialogueTurn] = Field(default_factory=list)
    active_edges: list[TemporalEdge] = Field(default_factory=list)
    historical_edges: list[TemporalEdge] = Field(default_factory=list)
    multi_hop_paths: list[list[TemporalEdge]] = Field(default_factory=list)
    episodic_hits: list[tuple[EpisodicChunk, float]] = Field(default_factory=list)
    suppressed_conflicting_episodes: int = 0
    formatted_prompt_context: str = ""
    prompt_tokens: int = 0
    latency_ms: float = 0.0
