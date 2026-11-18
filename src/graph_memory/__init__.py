"""Hierarchical & Temporal Graph Long-Term Memory Engine (`graph_memory`)."""

from graph_memory.conflict_resolver import (
    ADDITIVE_PREDICATES,
    FUNCTIONAL_PREDICATES,
    RETRACTION_TARGET_MAP,
    ConflictResolver,
)
from graph_memory.embeddings import EmbeddingService
from graph_memory.engine import HierarchicalGraphMemoryEngine, estimate_tokens
from graph_memory.episodic_store import EpisodicVectorStore
from graph_memory.extractor import ContextGroundedReader, MemoryFactExtractor
from graph_memory.forgetting import ForgettingEngine
from graph_memory.models import (
    ConflictResolutionEvent,
    ConsolidationReport,
    DialogueTurn,
    EntityNode,
    EntityType,
    EpisodicChunk,
    ExtractedFact,
    PredicateCategory,
    QueryIntent,
    RetrievalResult,
    TemporalEdge,
)
from graph_memory.temporal_graph import TemporalKnowledgeGraph
from graph_memory.working_memory import WorkingMemory

__all__ = [
    "ADDITIVE_PREDICATES",
    "FUNCTIONAL_PREDICATES",
    "RETRACTION_TARGET_MAP",
    "ConflictResolutionEvent",
    "ConflictResolver",
    "ConsolidationReport",
    "ContextGroundedReader",
    "DialogueTurn",
    "EmbeddingService",
    "EntityNode",
    "EntityType",
    "EpisodicChunk",
    "EpisodicVectorStore",
    "ExtractedFact",
    "ForgettingEngine",
    "HierarchicalGraphMemoryEngine",
    "MemoryFactExtractor",
    "PredicateCategory",
    "QueryIntent",
    "RetrievalResult",
    "TemporalEdge",
    "TemporalKnowledgeGraph",
    "WorkingMemory",
    "estimate_tokens",
]
