"""Dense vector embedding service with offline scikit-learn TF-IDF + SVD and optional OpenAI API."""

from __future__ import annotations

import os
from typing import Sequence

import httpx
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

# Rich semantic anchor corpus so offline TF-IDF + TruncatedSVD creates a fixed,
# deterministic dense embedding subspace persisted cleanly across SQLite sessions.
_SEMANTIC_ANCHOR_CORPUS: list[str] = [
    # Location / Residence domain
    "city live lives_in living residence home located relocated moved moving apartment",
    "beijing hangzhou shanghai shenzhen guangzhou nanjing chengdu wuhan suzhou tokyo singapore",
    "currently living in beijing moved to hangzhou relocated to shanghai based in shenzhen",
    "which city does user currently live in where is residence located prior city before moving",
    # Employment / Organization / Affiliation
    "works_at company lab institute university organization affiliation employed joined team",
    "bytedance alibaba tencent deepseek shanghai ai lab zhejiang lab tsinghua air msra sensetime",
    "stanford ai lab mit csail berkeley bair huggingface anthropic google deepmind",
    "which organization lab or company does collaborator or user work at affiliation",
    # Role / Title
    "current_role job title position occupation role intern junior engineer research scientist tech lead",
    "mlops intern backend engineer research intern staff engineer phd candidate principal investigator",
    "what is current role title position at company or research lab",
    # Database & Storage tech stack
    "primary_db database storage engine sql nosql vector store relational db backend database",
    "postgresql postgres mysql sqlite mongodb redis clickhouse milvus qdrant duckdb neo4j",
    "switched primary database from mysql to postgresql migrated backend db from mongodb to clickhouse",
    "which primary database engine does project or user rely on currently",
    # Libraries / Frameworks / Languages
    "uses_library framework toolkit package sdk runtime deep learning library programming",
    "pytorch jax tensorflow triton vllm langchain llamaindex fastapi rust python cpp cuda ray",
    "no_longer_uses stopped using deprecated dropped removed library framework",
    "speaks_language natural language fluent mandarin english japanese german french cantonese",
    "completed_course university course curriculum class cs224n cs231n mit6824 cmu15445 cs336",
    # Projects & Collaborators
    "works_on project building developing leading repository system architecture codename",
    "project helios project atlas project nebula project aegis project kronos project aurora",
    "project vortex project prism project titan project zephyr",
    "collaborates_with partner teammate coauthor colleague advisor mentor collaborator",
    "dr mei zhou dr wei lin prof hao zhang dr yue tang drbohan liu dr qian wu",
    # Target Conferences & Research Venues
    "target_conference submission venue paper deadline academic conference workshop publication",
    "neurips icml iclr cvpr acl emnlp sigmod vldb osdi sosp aaai kdd",
    "retargeted paper submission conference from icml to neurips aiming for iclr",
    # Dietary Preferences & Lifestyle Constraints
    "dietary_preference food diet meal restriction allergy eating habit cuisine",
    "vegetarian vegan pescatarian lactose_free halal gluten_free keto low_carb omnivore",
    "what dietary preference or food restriction should we keep in mind",
    # Hardware & GPU Budget Constraints
    "gpu_budget compute constraint hardware vram cluster gpu allocation training budget",
    "single rtx 3090 24gb dual rtx 4090 48gb 4x a100 320gb 8x h100 640gb 2x l40s 96gb a6000",
    "upgraded gpu compute budget constraint hardware vram cluster allocation",
    # Temporal / Historical markers
    "before previously prior initially originally first earlier former history evolution timeline",
    "how did state change over time transition sequence across sessions",
    # General conversational / chitchat anchors for separation
    "weather rain sunny coffee espresso latte lunch ramen noodles campus gym workout",
    "mechanical keyboard switches tactile linear debugging segfault unit test git merge",
    "weekend movie hiking music playlist tired sleepy morning standup meeting",
]

# Expand anchor diversity with synthetic combinations for full rank >= 128
for _idx in range(120):
    _a = _SEMANTIC_ANCHOR_CORPUS[_idx % len(_SEMANTIC_ANCHOR_CORPUS)]
    _b = _SEMANTIC_ANCHOR_CORPUS[(_idx * 7 + 3) % len(_SEMANTIC_ANCHOR_CORPUS)]
    _SEMANTIC_ANCHOR_CORPUS.append(f"{_a} | {_b} #{_idx}")

_CACHED_OFFLINE_PIPELINE: tuple[FeatureUnion, TruncatedSVD] | None = None


def _get_offline_pipeline(dim: int = 128) -> tuple[FeatureUnion, TruncatedSVD]:
    """Build or return cached TF-IDF (word + char_wb) + TruncatedSVD projection."""
    global _CACHED_OFFLINE_PIPELINE
    if _CACHED_OFFLINE_PIPELINE is not None and _CACHED_OFFLINE_PIPELINE[1].n_components == dim:
        return _CACHED_OFFLINE_PIPELINE

    features = FeatureUnion(
        transformer_list=[
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    lowercase=True,
                    sublinear_tf=True,
                    max_features=4096,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    lowercase=True,
                    sublinear_tf=True,
                    max_features=4096,
                ),
            ),
        ]
    )
    tfidf_matrix = features.fit_transform(_SEMANTIC_ANCHOR_CORPUS)
    max_comp = min(dim, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=max_comp, random_state=42)
    svd.fit(tfidf_matrix)
    if dim == 128 and max_comp == 128:
        _CACHED_OFFLINE_PIPELINE = (features, svd)
    return features, svd


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    safe_norms = np.where(norms > 1e-9, norms, 1.0)
    out = mat / safe_norms
    # Fallback deterministic unit vector if input text was completely empty
    zero_rows = (norms.squeeze(-1) <= 1e-9)
    if np.any(zero_rows):
        out[zero_rows, 0] = 1.0
    return out.astype(np.float32)


class EmbeddingService:
    """Dual-mode normalized dense vector embedder (Offline TF-IDF+SVD or OpenAI /v1/embeddings)."""

    def __init__(
        self,
        dim: int = 128,
        use_openai: bool = False,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.dim = dim
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.use_openai = bool(use_openai and self.api_key)
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.model = model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self._features, self._svd = _get_offline_pipeline(dim=dim)

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Synchronously compute L2-normalized dense embeddings of shape (N, dim)."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        tfidf = self._features.transform(list(texts))
        projected = self._svd.transform(tfidf)
        if projected.shape[1] < self.dim:
            pad = np.zeros((projected.shape[0], self.dim - projected.shape[1]), dtype=np.float32)
            projected = np.hstack([projected, pad])
        return _l2_normalize(projected)

    def embed_text(self, text: str) -> np.ndarray:
        """Synchronously compute a single 1D L2-normalized embedding vector."""
        return self.embed_batch([text])[0]

    async def aembed_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Asynchronously compute embeddings via OpenAI API if enabled, else offline SVD."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if not self.use_openai:
            return self.embed_batch(texts)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, object] = {
            "model": self.model,
            "input": list(texts),
        }
        if "text-embedding-3" in self.model:
            payload["dimensions"] = self.dim

        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(f"{self.base_url}/embeddings", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()["data"]
            data_sorted = sorted(data, key=lambda item: item["index"])
            vecs = np.array([item["embedding"] for item in data_sorted], dtype=np.float32)
            return _l2_normalize(vecs)

    async def aembed_text(self, text: str) -> np.ndarray:
        vecs = await self.aembed_batch([text])
        return vecs[0]
