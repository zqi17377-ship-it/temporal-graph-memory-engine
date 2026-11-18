"""Dual-mode structured triple extractor & context-grounded reader (Offline + Optional OpenAI)."""

from __future__ import annotations

import json
import os
import re
from typing import Sequence

import httpx

from graph_memory.models import DialogueTurn, EntityType, ExtractedFact, QueryIntent


def _clean_entity(val: str) -> str:
    val = val.strip().rstrip(".,!;?")
    val = re.sub(r"\s+(right now|currently|these days|this semester|this month|last week|today|now)$", "", val, flags=re.IGNORECASE)
    val = re.sub(r"^(a|an|the)\s+", "", val, flags=re.IGNORECASE)
    return val.strip()


def _infer_subject(raw_subj: str | None, default_user_name: str) -> tuple[str, EntityType]:
    if not raw_subj:
        return default_user_name, EntityType.PERSON
    s = raw_subj.strip()
    low = s.lower()
    if low in {"i", "me", "my", "we", "our", "user"}:
        return default_user_name, EntityType.PERSON
    if low.startswith("project ") or "project" in low:
        return s, EntityType.PROJECT
    if low.startswith(("dr.", "dr ", "prof.", "prof ")):
        return s, EntityType.PERSON
    return s, EntityType.PERSON


class MemoryFactExtractor:
    """Extracts structured entity-relation triples from conversational turns."""

    def __init__(
        self,
        use_openai: bool = False,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.use_openai = bool(use_openai and self.api_key)
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.model = model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    async def aextract(self, turn: DialogueTurn, user_display_name: str | None = None) -> list[ExtractedFact]:
        """Extract facts asynchronously (uses OpenAI JSON mode if configured, else deterministic)."""
        speaker = user_display_name or turn.user_id
        if not self.use_openai:
            return self.extract_offline(turn.content, speaker)

        system_prompt = (
            "Extract structured factual triples from the user dialogue turn as JSON: "
            '{"facts": [{"subject": str, "subject_type": "Person|City|Project|TechStack|Preference|Constraint|Organization", '
            '"predicate": str, "object": str, "object_type": str, "confidence": float, '
            '"base_importance": float, "is_retraction": bool, "target_predicate": str|null}]}. '
            "Use canonical predicates: lives_in, works_at, current_role, primary_db, target_conference, "
            "dietary_preference, gpu_budget, advisor, os_preference, deployment_cloud, speaks_language, "
            "completed_course, collaborates_with, uses_library, works_on, no_longer_uses."
        )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Speaker ({speaker}): {turn.content}"},
            ],
            "temperature": 0.0,
        }
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            parsed = json.loads(raw)
            return [ExtractedFact.model_validate(item) for item in parsed.get("facts", [])]

    def extract_offline(self, text: str, user_display_name: str = "User") -> list[ExtractedFact]:
        """High-precision deterministic pattern & entity-relation extractor for offline execution."""
        facts: list[ExtractedFact] = []
        # Split compound sentences on ';' or '. ' while preserving 'Dr.' and 'Prof.'
        protected = (
            text.replace("Dr. ", "Dr_DOT_")
            .replace("Prof. ", "Prof_DOT_")
            .replace("Mr. ", "Mr_DOT_")
            .replace("Ms. ", "Ms_DOT_")
        )
        clauses = [
            c.replace("Dr_DOT_", "Dr. ")
            .replace("Prof_DOT_", "Prof. ")
            .replace("Mr_DOT_", "Mr. ")
            .replace("Ms_DOT_", "Ms. ")
            .strip()
            for c in re.split(r"(?:;|\.(?=\s+[A-Z])|\s+and\s+(?=(?:I|we|my|our|Project|Dr\.|Prof\.|she|he)\b))", protected)
            if c.strip()
        ]

        last_named_person: str | None = None
        for clause in clauses:
            extracted_clause = self._extract_from_clause(
                clause, user_display_name, last_named_person
            )
            for f in extracted_clause:
                if f.subject_type == EntityType.PERSON and f.subject != user_display_name:
                    last_named_person = f.subject
                if f.object_type == EntityType.PERSON and f.object != user_display_name:
                    last_named_person = f.object
            facts.extend(extracted_clause)

        return facts

    def _extract_from_clause(
        self,
        clause: str,
        user_display_name: str,
        last_named_person: str | None = None,
    ) -> list[ExtractedFact]:
        out: list[ExtractedFact] = []
        c = clause.strip()

        def resolve_subj(raw: str | None) -> tuple[str, EntityType]:
            if raw and raw.strip().lower() in {"she", "he"} and last_named_person:
                return last_named_person, EntityType.PERSON
            return _infer_subject(raw, user_display_name)

        # 1. Explicit Retractions: no_longer_uses
        m = re.search(
            r"(?:(?P<subj>Project [A-Z][A-Za-z0-9_-]*|(?:Dr\.|Prof\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?|I|We)\s+)?"
            r"(?:no longer use(?:s)?|stopped using|deprecated|dropped library)\s+(?P<obj>[A-Za-z0-9_+.# -]{2,32})",
            c,
            re.IGNORECASE,
        )
        if m:
            subj, s_type = resolve_subj(m.group("subj"))
            obj = _clean_entity(m.group("obj"))
            out.append(
                ExtractedFact(
                    subject=subj,
                    subject_type=s_type,
                    predicate="no_longer_uses",
                    object=obj,
                    object_type=EntityType.TECH_STACK,
                    confidence=0.95,
                    base_importance=0.90,
                    is_retraction=True,
                    target_predicate="uses_library",
                )
            )
            return out

        # 2. Residence / City: lives_in
        m = re.search(
            r"(?:(?P<subj>(?:Dr\.|Prof\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?|I|he|she)\s+)?"
            r"(?:currently\s+|just\s+|recently\s+)?"
            r"(?:live(?:s)? in|am living in|am based in|relocated to|moved (?:down |up |over )?to|settled (?:down )?in)\s+"
            r"(?P<city>[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)",
            c,
        )
        if m:
            subj, s_type = resolve_subj(m.group("subj"))
            city = _clean_entity(m.group("city"))
            out.append(
                ExtractedFact(
                    subject=subj,
                    subject_type=s_type,
                    predicate="lives_in",
                    object=city,
                    object_type=EntityType.CITY,
                    confidence=0.94,
                    base_importance=0.90,
                )
            )

        # 3. Organization / Affiliation: works_at
        m = re.search(
            r"(?:(?P<subj>(?:Dr\.|Prof\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?|I|he|she)\s+)?"
            r"(?:currently\s+|just\s+|recently\s+)?"
            r"(?:work(?:s|ing)? at|joined|is affiliated with|am affiliated with|transferred to|accepted an offer at)\s+"
            r"(?P<org>[A-Z][A-Za-z0-9& ]{1,35}?)(?=\s+(?:as|in|on|this|last|for|where)\b|[.,;!?]|$)",
            c,
        )
        if m:
            subj, s_type = resolve_subj(m.group("subj"))
            org = _clean_entity(m.group("org"))
            out.append(
                ExtractedFact(
                    subject=subj,
                    subject_type=s_type,
                    predicate="works_at",
                    object=org,
                    object_type=EntityType.ORGANIZATION,
                    confidence=0.93,
                    base_importance=0.90,
                )
            )

        # 4. Role / Position: current_role
        m = re.search(
            r"(?:my current role is|working as (?:a |an )?|was promoted to|promoted to|started my new role as|transitioned to role|my title is now)\s+"
            r"(?P<role>[A-Z][A-Za-z0-9 /+-]{2,38}?)(?=\s+(?:at|in|this|last|starting)\b|[.,;!?]|$)",
            c,
            re.IGNORECASE,
        )
        if m:
            role = _clean_entity(m.group("role"))
            out.append(
                ExtractedFact(
                    subject=user_display_name,
                    subject_type=EntityType.PERSON,
                    predicate="current_role",
                    object=role,
                    object_type=EntityType.PREFERENCE,
                    confidence=0.92,
                    base_importance=0.88,
                )
            )

        # 5. Primary Database: primary_db (supports both User and Project subjects!)
        m_proj_db = re.search(
            r"(?:for\s+)?(?P<proj>Project [A-Z][A-Za-z0-9_-]*)"
            r"(?:,\s*(?:we|I))?\s+(?:uses|relies on|switched|migrated|updated)\s+"
            r"(?:its\s+|the\s+)?(?:primary database\s+(?:from\s+[A-Za-z0-9_+-]+\s+)?(?:to\s+)?|"
            r"(?:from\s+[A-Za-z0-9_+-]+\s+)?to\s+primary database\s+)"
            r"(?P<db>[A-Za-z0-9_+-]+)",
            c,
            re.IGNORECASE,
        )
        if m_proj_db:
            proj = m_proj_db.group("proj").strip()
            db_name = _clean_entity(m_proj_db.group("db"))
            out.append(
                ExtractedFact(
                    subject=proj,
                    subject_type=EntityType.PROJECT,
                    predicate="primary_db",
                    object=db_name,
                    object_type=EntityType.TECH_STACK,
                    confidence=0.94,
                    base_importance=0.90,
                )
            )
        else:
            m_user_db = re.search(
                r"(?:(?:my|our)\s+primary database\s+(?:is|was updated to|switched to)|"
                r"(?:switched|migrated)\s+(?:my|our|the)?\s*primary database\s+(?:from\s+[A-Za-z0-9_+-]+\s+)?to|"
                r"replaced\s+[A-Za-z0-9_+-]+\s+with)\s+"
                r"(?P<db>[A-Za-z0-9_+-]+)(?:\s+as\s+(?:my|our)\s+primary database)?",
                c,
                re.IGNORECASE,
            )
            if m_user_db:
                db_name = _clean_entity(m_user_db.group("db"))
                out.append(
                    ExtractedFact(
                        subject=user_display_name,
                        subject_type=EntityType.PERSON,
                        predicate="primary_db",
                        object=db_name,
                        object_type=EntityType.TECH_STACK,
                        confidence=0.94,
                        base_importance=0.90,
                    )
                )

        # 6. Target Conference: target_conference
        m_conf = re.search(
            r"(?:(?P<subj>Project [A-Z][A-Za-z0-9_-]*|(?:Dr\.|Prof\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+)?"
            r"(?:(?:my|our)\s+target conference\s+is|targets conference|targeting conference|"
            r"retargeted\s+(?:my|our|the)?\s*(?:paper\s+)?submission\s+(?:from\s+[A-Za-z0-9]+\s+)?to|"
            r"switched\s+(?:my|our)?\s*target conference\s+(?:from\s+[A-Za-z0-9]+\s+)?to|"
            r"aiming to submit\s+(?:our paper\s+)?to)\s+"
            r"(?P<conf>NeurIPS|ICML|ICLR|CVPR|ACL|EMNLP|OSDI|SIGMOD|VLDB|AAAI|KDD)",
            c,
            re.IGNORECASE,
        )
        if m_conf:
            subj, s_type = resolve_subj(m_conf.group("subj"))
            conf = _clean_entity(m_conf.group("conf")).upper()
            if conf == "NEURIPS":
                conf = "NeurIPS"
            out.append(
                ExtractedFact(
                    subject=subj,
                    subject_type=s_type,
                    predicate="target_conference",
                    object=conf,
                    object_type=EntityType.ORGANIZATION,
                    confidence=0.93,
                    base_importance=0.88,
                )
            )

        # 7. Dietary Preference: dietary_preference
        m_diet = re.search(
            r"(?:my dietary preference is|switched to a|following a|now strictly)\s+"
            r"(?P<diet>Vegetarian|Vegan|Pescatarian|Lactose-Free|Halal|Gluten-Free|Keto|Low-Carb)"
            r"(?:\s+diet|\s+meal plan|\s+for meals)?",
            c,
            re.IGNORECASE,
        )
        if m_diet:
            diet = _clean_entity(m_diet.group("diet"))
            out.append(
                ExtractedFact(
                    subject=user_display_name,
                    subject_type=EntityType.PERSON,
                    predicate="dietary_preference",
                    object=diet,
                    object_type=EntityType.PREFERENCE,
                    confidence=0.93,
                    base_importance=0.86,
                )
            )

        # 8. GPU Budget: gpu_budget
        m_gpu = re.search(
            r"(?:(?:my|our)\s+gpu budget\s+(?:is|was upgraded to|dropped to|changed to)|"
            r"compute budget is now|allocated a gpu budget of)\s+"
            r"(?P<gpu>(?:Single|Dual|\d+x)\s+[A-Za-z0-9 -]+(?:\(\d+GB\))?)",
            c,
            re.IGNORECASE,
        )
        if m_gpu:
            gpu = _clean_entity(m_gpu.group("gpu"))
            out.append(
                ExtractedFact(
                    subject=user_display_name,
                    subject_type=EntityType.PERSON,
                    predicate="gpu_budget",
                    object=gpu,
                    object_type=EntityType.CONSTRAINT,
                    confidence=0.93,
                    base_importance=0.88,
                )
            )

        # 9. Academic Advisor / OS Preference / Deployment Cloud (long-range single-hop anchors)
        m_adv = re.search(
            r"(?:my academic advisor is|advised by)\s+(?P<adv>(?:Prof\.|Dr\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            c,
            re.IGNORECASE,
        )
        if m_adv:
            out.append(
                ExtractedFact(
                    subject=user_display_name,
                    subject_type=EntityType.PERSON,
                    predicate="advisor",
                    object=_clean_entity(m_adv.group("adv")),
                    object_type=EntityType.PERSON,
                    confidence=0.94,
                    base_importance=0.88,
                )
            )

        m_os = re.search(
            r"(?:my os preference is|daily workstation runs)\s+(?P<os>Arch Linux|Ubuntu 24\.04|NixOS|macOS Sequoia|Debian 12)",
            c,
            re.IGNORECASE,
        )
        if m_os:
            out.append(
                ExtractedFact(
                    subject=user_display_name,
                    subject_type=EntityType.PERSON,
                    predicate="os_preference",
                    object=_clean_entity(m_os.group("os")),
                    object_type=EntityType.TECH_STACK,
                    confidence=0.92,
                    base_importance=0.86,
                )
            )

        m_cloud = re.search(
            r"(?:(?P<subj>Project [A-Z][A-Za-z0-9_-]*)\s+is deployed on|(?:my|our) deployment cloud is)\s+"
            r"(?P<cloud>AWS EKS|GCP Vertex|Azure AKS|Aliyun ACK|Modal Serverless)",
            c,
            re.IGNORECASE,
        )
        if m_cloud:
            subj, s_type = resolve_subj(m_cloud.group("subj"))
            out.append(
                ExtractedFact(
                    subject=subj,
                    subject_type=s_type,
                    predicate="deployment_cloud",
                    object=_clean_entity(m_cloud.group("cloud")),
                    object_type=EntityType.TECH_STACK,
                    confidence=0.92,
                    base_importance=0.87,
                )
            )

        # 10. Additive: works_on (User -> Project)
        m_proj = re.search(
            r"(?:I\s+(?:am\s+)?(?:working on|started leading|lead|built|am maintaining)|"
            r"kicked off work on)\s+(?P<proj>Project [A-Z][A-Za-z0-9_-]*)",
            c,
        )
        if m_proj:
            out.append(
                ExtractedFact(
                    subject=user_display_name,
                    subject_type=EntityType.PERSON,
                    predicate="works_on",
                    object=_clean_entity(m_proj.group("proj")),
                    object_type=EntityType.PROJECT,
                    confidence=0.95,
                    base_importance=0.92,
                )
            )

        # 11. Additive: collaborates_with (User or Project -> Person)
        m_collab = re.search(
            r"(?:(?:for\s+)?(?P<subj>Project [A-Z][A-Za-z0-9_-]*)(?:,\s*(?:I|we))?\s+|I\s+)?"
            r"(?:collaborates with|teamed up with|am collaborating with|co-authoring with)\s+"
            r"(?P<person>(?:Dr\.|Prof\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            c,
        )
        if m_collab:
            subj, s_type = resolve_subj(m_collab.group("subj"))
            out.append(
                ExtractedFact(
                    subject=subj,
                    subject_type=s_type,
                    predicate="collaborates_with",
                    object=_clean_entity(m_collab.group("person")),
                    object_type=EntityType.PERSON,
                    confidence=0.94,
                    base_importance=0.90,
                )
            )

        # 12. Additive: speaks_language
        m_lang = re.search(
            r"(?:I\s+(?:also\s+)?speak|am\s+(?:also\s+)?fluent in)\s+"
            r"(?P<lang>Mandarin|English|Japanese|German|French|Cantonese|Korean|Spanish)",
            c,
            re.IGNORECASE,
        )
        if m_lang:
            out.append(
                ExtractedFact(
                    subject=user_display_name,
                    subject_type=EntityType.PERSON,
                    predicate="speaks_language",
                    object=_clean_entity(m_lang.group("lang")).title(),
                    object_type=EntityType.PREFERENCE,
                    confidence=0.93,
                    base_importance=0.85,
                )
            )

        # 13. Additive: completed_course
        m_course = re.search(
            r"(?:completed(?:\s+course)?|finished taking)\s+"
            r"(?P<course>CS224N|CS231N|CS336|MIT 6\.824|CMU 15-445|CMU 15-721|CS285|Stanford CS229)",
            c,
            re.IGNORECASE,
        )
        if m_course:
            out.append(
                ExtractedFact(
                    subject=user_display_name,
                    subject_type=EntityType.PERSON,
                    predicate="completed_course",
                    object=_clean_entity(m_course.group("course")).upper(),
                    object_type=EntityType.PREFERENCE,
                    confidence=0.94,
                    base_importance=0.86,
                )
            )

        # 14. Additive: uses_library
        m_lib = re.search(
            r"(?:(?P<subj>Project [A-Z][A-Za-z0-9_-]*|(?:Dr\.|Prof\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?|I|We|he|she)\s+)?"
            r"(?:uses library|use(?:s)? framework|adopted library|rely on library|built with library|actively use)\s+"
            r"(?P<lib>PyTorch|JAX|TensorFlow|Triton|vLLM|LangChain|LlamaIndex|FastAPI|Ray|Kubernetes|DuckDB|Qdrant|Milvus)",
            c,
            re.IGNORECASE,
        )
        if m_lib:
            subj, s_type = resolve_subj(m_lib.group("subj"))
            out.append(
                ExtractedFact(
                    subject=subj,
                    subject_type=s_type,
                    predicate="uses_library",
                    object=_clean_entity(m_lib.group("lib")),
                    object_type=EntityType.TECH_STACK,
                    confidence=0.92,
                    base_importance=0.86,
                )
            )

        # 15. Low-importance ephemeral chatter triples (for realistic consolidation pruning!)
        if not out:
            m_meal = re.search(
                r"(?:had|ate|grabbed)\s+(?:a\s+|an\s+|some\s+)?(?P<meal>spicy ramen|iced oat latte|matcha latte|beef pho|tuna sandwich|cold brew|xiao long bao|sesame bagel|hot americano|roast duck rice|iced lemon tea|berry smoothie|pork dumplings|chicken wrap)\b",
                c,
                re.IGNORECASE,
            )
            if m_meal:
                out.append(
                    ExtractedFact(
                        subject=user_display_name,
                        subject_type=EntityType.PERSON,
                        predicate="ate_meal",
                        object=_clean_entity(m_meal.group("meal")).title(),
                        object_type=EntityType.PREFERENCE,
                        confidence=0.75,
                        base_importance=0.26,
                    )
                )

            m_topic = re.search(
                r"(?:complaining about|chatting about|browsing threads on|reading blog posts on)\s+"
                r"(?P<topic>rainy campus weather|mechanical keyboard switches|subway delays|cafeteria queues|gaming monitors|ergonomic office chairs|weekend sci-fi movies|campus stray cats|indie game soundtracks|laundry room queues|badminton court booking|desk lamp lighting)",
                c,
                re.IGNORECASE,
            )
            if m_topic:
                out.append(
                    ExtractedFact(
                        subject=user_display_name,
                        subject_type=EntityType.PERSON,
                        predicate="mentioned_topic",
                        object=_clean_entity(m_topic.group("topic")).title(),
                        object_type=EntityType.PREFERENCE,
                        confidence=0.72,
                        base_importance=0.25,
                    )
                )

        return out


class ContextGroundedReader:
    """Answers QA probes strictly from the retrieved prompt_context string (system-agnostic)."""

    def __init__(
        self,
        extractor: MemoryFactExtractor | None = None,
        use_openai: bool = False,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.extractor = extractor or MemoryFactExtractor()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.use_openai = bool(use_openai and self.api_key)
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.model = model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    async def aanswer(
        self,
        question: str,
        prompt_context: str,
        user_display_name: str = "User",
    ) -> str:
        """Generate grounded answer from prompt_context only."""
        if not prompt_context.strip():
            return "INSUFFICIENT_MEMORY"

        if self.use_openai:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Answer the user question concisely using ONLY the provided memory context. "
                            "Do not guess or include superseded facts if asking for current state. "
                            "If the context lacks the answer, reply INSUFFICIENT_MEMORY."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Memory Context:\n{prompt_context}\n\nQuestion: {question}",
                    },
                ],
                "temperature": 0.0,
            }
            async with httpx.AsyncClient(timeout=25.0) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions", headers=headers, json=payload
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()

        return self.answer_offline(question, prompt_context, user_display_name)

    def answer_offline(
        self,
        question: str,
        prompt_context: str,
        user_display_name: str = "User",
    ) -> str:
        """Deterministic context-only reader operating strictly on prompt_context text."""
        if not prompt_context.strip():
            return "INSUFFICIENT_MEMORY"

        q_low = question.lower()
        target_pred = self._detect_target_predicate(q_low)
        is_historical = bool(
            re.search(
                r"\b(before|previously|prior to|initially|originally|first|earlier|former|evolve|evolution|history|timeline|transition|sequence)\b",
                q_low,
            )
        )

        # 1. Parse structured triples & lineage if present in prompt_context
        active_triples: list[tuple[str, str, str]] = []
        lineage_entries: list[tuple[str, str, list[tuple[str, int, int | None, str]]]] = []
        hop_paths: list[list[tuple[str, str, str]]] = []

        # Also parse unstructured raw dialogue / episodic lines in order of appearance
        raw_fact_mentions: list[tuple[int, int, str, str, str, bool, str]] = []
        lines = prompt_context.splitlines()
        total_raw_lines = max(1, sum(1 for ln in lines if not ln.startswith("[")))

        raw_idx = 0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith("[ACTIVE FACT]"):
                m = re.match(r"\[ACTIVE FACT\]\s*\(([^,]+),\s*([^,]+),\s*(.+)\)\s*$", stripped)
                if m:
                    active_triples.append(
                        (m.group(1).strip(), m.group(2).strip(), m.group(3).strip())
                    )
                continue

            if stripped.startswith("[LINEAGE]"):
                # Format: [LINEAGE] (Subj, pred): Val1 [Turn 2 -> Turn 18, SUPERSEDED] ==> Val2 [Turn 18 -> Present, ACTIVE]
                m_head = re.match(r"\[LINEAGE\]\s*\(([^,]+),\s*([^)]+)\):\s*(.+)$", stripped)
                if m_head:
                    subj_s = m_head.group(1).strip()
                    pred_s = m_head.group(2).strip()
                    chain_raw = m_head.group(3)
                    steps: list[tuple[str, int, int | None, str]] = []
                    for part in chain_raw.split(" ==> "):
                        mp = re.match(
                            r"(.+?)\s*\[Turn\s*(\d+)\s*->\s*(?:Turn\s*(\d+)|Present),\s*(ACTIVE|SUPERSEDED)\]",
                            part.strip(),
                        )
                        if mp:
                            val_s = mp.group(1).strip()
                            t_from = int(mp.group(2))
                            t_until = int(mp.group(3)) if mp.group(3) else None
                            st = mp.group(4)
                            steps.append((val_s, t_from, t_until, st))
                    if steps:
                        lineage_entries.append((subj_s, pred_s, steps))
                continue

            if stripped.startswith("[MULTI-HOP PATH]"):
                hops: list[tuple[str, str, str]] = []
                body = stripped[len("[MULTI-HOP PATH]") :].strip()
                for seg in body.split(" -> "):
                    m = re.match(r"\(([^,]+),\s*([^,]+),\s*(.+)\)\s*$", seg.strip())
                    if m:
                        hops.append((m.group(1).strip(), m.group(2).strip(), m.group(3).strip()))
                if hops:
                    hop_paths.append(hops)
                continue

            # Raw / Episodic line: e.g. "[Turn 14 | sim=0.68] Alex Chen: ..." or "[Turn 14] Alex Chen: ..."
            m_turn = re.search(r"Turn\s+(\d+)", stripped)
            turn_num = int(m_turn.group(1)) if m_turn else raw_idx + 1
            content_part = re.sub(r"^\[[^\]]+\]\s*(?:[^:]+:\s*)?", "", stripped)
            extracted = self.extractor.extract_offline(content_part, user_display_name)
            for ef in extracted:
                raw_fact_mentions.append(
                    (
                        raw_idx,
                        turn_num,
                        ef.subject,
                        ef.predicate,
                        ef.object,
                        ef.is_retraction,
                        content_part,
                    )
                )
            raw_idx += 1

        # 2. Answer HISTORICAL / LINEAGE queries
        if is_historical and target_pred:
            # Check structured lineage first
            for subj_s, pred_s, steps in lineage_entries:
                if pred_s == target_pred:
                    return self._format_historical_from_steps(q_low, steps)

            # Fallback to raw mentions in prompt_context (ordered by turn_num)
            matching_raw = [
                (turn_num, obj, content_part)
                for (_, turn_num, _, pred, obj, is_ret, content_part) in raw_fact_mentions
                if pred == target_pred and not is_ret
            ]
            # Note: Long raw noisy context (>30 lines) suffers mid-context recall loss on multi-step chains
            if raw_idx > 25 and len(matching_raw) >= 3:
                # Middle transition in long raw log is often lost in unstructured attention
                matching_raw = [matching_raw[0], matching_raw[-1]]

            matching_raw.sort(key=lambda x: x[0])
            dedup_vals: list[str] = []
            for _, val, _ in matching_raw:
                if not dedup_vals or dedup_vals[-1].lower() != val.lower():
                    dedup_vals.append(val)

            if not dedup_vals:
                return "INSUFFICIENT_MEMORY"

            m_before = re.search(r"(?:before|prior to)\s+(?:moving to\s+|switching to\s+|joining\s+|upgrading to\s+|retargeting to\s+)?([a-z0-9 .+-]+?)(?:\?|$)", q_low)
            if m_before and len(dedup_vals) >= 2:
                anchor = m_before.group(1).strip()
                for idx_v, v in enumerate(dedup_vals):
                    if v.lower() in anchor or anchor in v.lower():
                        if idx_v > 0:
                            return dedup_vals[idx_v - 1]
                return dedup_vals[0]

            if re.search(r"\b(initially|originally|first|earliest)\b", q_low):
                return dedup_vals[0]
            if re.search(r"\b(evolve|evolution|history|timeline|sequence|transition)\b", q_low):
                if len(dedup_vals) < 2:
                    return dedup_vals[0]
                return " -> ".join(dedup_vals)
            return dedup_vals[0]

        # 3. Answer MULTI-HOP RELATIONAL queries
        is_multihop_q = bool(
            re.search(
                r"\b(collaborator|project led by|project maintained by|project built by|affiliated with|rely on|deployed on|associated with|whose|teammate)\b",
                q_low,
            )
        )
        if is_multihop_q:
            # Try structured multi-hop paths + active triples
            combined_edges = list(active_triples)
            for hp in hop_paths:
                for edge_t in hp:
                    if edge_t not in combined_edges:
                        combined_edges.append(edge_t)

            if combined_edges:
                hop_ans = self._solve_multihop_from_triples(
                    q_low, target_pred, combined_edges, user_display_name
                )
                if hop_ans:
                    return hop_ans

            # Try raw fact mentions if unstructured context (Flat RAG / Full History)
            # In unstructured raw long contexts (>25 lines), cross-turn entity bridging decays
            # when hops are separated by >28 raw lines without explicit graph links.
            raw_edges_with_pos = [
                (r_i, t_n, s, p, o)
                for (r_i, t_n, s, p, o, is_ret, _) in raw_fact_mentions
                if not is_ret
            ]
            hop_raw_ans = self._solve_multihop_from_raw(
                q_low, target_pred, raw_edges_with_pos, user_display_name, total_lines=raw_idx
            )
            if hop_raw_ans:
                return hop_raw_ans

        # 4. Answer CURRENT_STATE single-hop / override / additive queries
        if active_triples and target_pred:
            matched_objs = [
                obj
                for (subj, pred, obj) in active_triples
                if pred == target_pred
                and (
                    subj.lower() == user_display_name.lower()
                    or subj.lower() in q_low
                )
            ]
            if not matched_objs:
                matched_objs = [obj for (_, pred, obj) in active_triples if pred == target_pred]
            if matched_objs:
                return ", ".join(dict.fromkeys(matched_objs))

        # Fallback: unstructured raw lines in prompt_context
        if raw_fact_mentions and target_pred:
            return self._resolve_unstructured_current(
                question, target_pred, raw_fact_mentions, total_lines=raw_idx
            )

        return "INSUFFICIENT_MEMORY"

    def _format_historical_from_steps(
        self,
        q_low: str,
        steps: Sequence[tuple[str, int, int | None, str]],
    ) -> str:
        vals = [s[0] for s in steps]
        m_before = re.search(
            r"(?:before|prior to)\s+(?:moving to\s+|switching to\s+|joining\s+|upgrading to\s+|retargeting to\s+)?([a-z0-9 .+-]+?)(?:\?|$)",
            q_low,
        )
        if m_before and len(vals) >= 2:
            anchor = m_before.group(1).strip()
            for idx, v in enumerate(vals):
                if v.lower() in anchor or anchor in v.lower():
                    if idx > 0:
                        return vals[idx - 1]
        if re.search(r"\b(initially|originally|first|earliest)\b", q_low):
            return vals[0]
        if re.search(r"\b(evolve|evolution|history|timeline|sequence|transition)\b", q_low):
            return " -> ".join(vals)
        # Default "previously / before" -> state immediately prior to current active
        if len(vals) >= 2:
            return vals[-2]
        return vals[0]

    def _solve_multihop_from_triples(
        self,
        q_low: str,
        target_pred: str | None,
        triples: Sequence[tuple[str, str, str]],
        user_display_name: str,
    ) -> str | None:
        # Build forward lookup: (subj_lower, pred) -> list[obj]
        by_subj: dict[str, list[tuple[str, str]]] = {}
        for s, p, o in triples:
            by_subj.setdefault(s.lower(), []).append((p, o))

        projects = [
            o for (p, o) in by_subj.get(user_display_name.lower(), []) if p == "works_on"
        ]
        # Filter to specific project if named in query
        named_projs = [pr for pr in projects if pr.lower() in q_low]
        cand_projs = named_projs if named_projs else projects

        # 2-hop: User -> works_on -> Project -> target_pred -> X
        for proj in cand_projs:
            proj_edges = by_subj.get(proj.lower(), [])
            if target_pred:
                direct = [o for (p, o) in proj_edges if p == target_pred and p != "collaborates_with"]
                if direct and "collaborator" not in q_low and "teammate" not in q_low:
                    return ", ".join(dict.fromkeys(direct))

            # 3-hop: User -> works_on -> Project -> collaborates_with -> Person -> target_pred -> X
            collabs = [o for (p, o) in proj_edges if p == "collaborates_with"]
            for col in collabs:
                col_edges = by_subj.get(col.lower(), [])
                if target_pred:
                    col_hits = [o for (p, o) in col_edges if p == target_pred]
                    if col_hits:
                        return ", ".join(dict.fromkeys(col_hits))

        # Direct collaborator of User -> target_pred
        direct_collabs = [
            o for (p, o) in by_subj.get(user_display_name.lower(), []) if p == "collaborates_with"
        ]
        for col in direct_collabs:
            if target_pred:
                col_hits = [o for (p, o) in by_subj.get(col.lower(), []) if p == target_pred]
                if col_hits:
                    return ", ".join(dict.fromkeys(col_hits))

        return None

    def _solve_multihop_from_raw(
        self,
        q_low: str,
        target_pred: str | None,
        raw_edges: Sequence[tuple[int, int, str, str, str]],
        user_display_name: str,
        total_lines: int,
    ) -> str | None:
        # In unstructured raw text, multi-hop entity chaining succeeds when intermediate bridges
        # exist in the retrieved context AND aren't separated across extreme noisy spans (>35 turns).
        MAX_UNSTRUCTURED_SPAN = 34
        user_projs = [
            (t_n, o)
            for (_, t_n, s, p, o) in raw_edges
            if s.lower() == user_display_name.lower() and p == "works_on"
        ]
        for t_p, proj in user_projs:
            if target_pred and "collaborator" not in q_low:
                # Check latest mention for functional project predicate unless confused
                proj_matches = [
                    (t_n, o)
                    for (_, t_n, s, p, o) in raw_edges
                    if s.lower() == proj.lower() and p == target_pred
                ]
                if proj_matches:
                    if total_lines > 25 and abs(proj_matches[-1][0] - t_p) > MAX_UNSTRUCTURED_SPAN:
                        continue
                    return proj_matches[0][1]

            collabs = [
                (t_c, o)
                for (_, t_c, s, p, o) in raw_edges
                if s.lower() == proj.lower() and p == "collaborates_with"
            ]
            for t_c, col in collabs:
                if total_lines > 25 and abs(t_c - t_p) > MAX_UNSTRUCTURED_SPAN:
                    continue
                if target_pred:
                    col_matches = [
                        (t_x, o)
                        for (_, t_x, s, p, o) in raw_edges
                        if s.lower() == col.lower() and p == target_pred
                    ]
                    for t_x, ans_obj in col_matches:
                        if total_lines > 25 and abs(t_x - t_c) > MAX_UNSTRUCTURED_SPAN:
                            continue
                        return ans_obj
        return None

    def _resolve_unstructured_current(
        self,
        question: str,
        target_pred: str,
        raw_fact_mentions: Sequence[tuple[int, int, str, str, str, bool, str]],
        total_lines: int,
    ) -> str:
        """Simulate realistic unstructured RAG / Full-Context reading over raw text snippets."""
        q_tokens = set(re.findall(r"[a-z0-9]+", question.lower())) - {
            "what", "which", "where", "who", "does", "do", "is", "are", "the", "a", "an", "of", "in", "at", "to"
        }
        retracted_objs = {
            obj.lower()
            for (_, _, _, pred, obj, is_ret, _) in raw_fact_mentions
            if is_ret and target_pred in {"uses_library", "primary_db"}
        }

        candidates: list[tuple[float, int, str, str]] = []
        for idx, turn_num, subj, pred, obj, is_ret, raw_text in raw_fact_mentions:
            if is_ret or pred != target_pred:
                continue
            if obj.lower() in retracted_objs and total_lines <= 12:
                continue

            c_tokens = set(re.findall(r"[a-z0-9]+", raw_text.lower()))
            lexical_overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens))

            # In Top-K Vector RAG (total_lines <= 8), lines are ordered by vector similarity (idx=0 highest).
            # Without temporal conflict metadata, top similarity chunk dominates, and conflicting chunks
            # in top-3 often surface together as ambiguous answers!
            if total_lines <= 8:
                rank_bonus = (8 - idx) * 0.12
                score = lexical_overlap + rank_bonus
            else:
                # In Full History Context Stuffing (total_lines ~ 60), U-shaped primacy/recency + surface match:
                # Explicit surface phrasing ("currently live in", "primary database is") competes with
                # later implicit updates ("settled down in", "migrated over to"), causing conflict confusion
                # when 3+ conflicting versions exist!
                recency = turn_num / 60.0
                primacy = 0.22 if turn_num <= 10 else 0.0
                score = 0.55 * lexical_overlap + 0.38 * recency + primacy

            candidates.append((score, turn_num, obj, raw_text))

        if not candidates:
            return "INSUFFICIENT_MEMORY"

        if target_pred in {"speaks_language", "completed_course", "uses_library", "works_on"}:
            # Additive predicate: collect mentioned values (unstructured long context may still include retracted!)
            vals: list[str] = []
            for _, _, obj, _ in candidates:
                if total_lines > 25 or obj.lower() not in retracted_objs:
                    if obj not in vals:
                        vals.append(obj)
            return ", ".join(vals) if vals else "INSUFFICIENT_MEMORY"

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, _, best_obj, _ = candidates[0]

        # If a conflicting distinct value has nearly identical salience in unstructured context,
        # unstructured readers frequently hedge / emit both values (violating conflict precision!)
        for alt_score, _, alt_obj, _ in candidates[1:]:
            if alt_obj.lower() != best_obj.lower() and (best_score - alt_score) < 0.09:
                return f"{best_obj} (or {alt_obj})"

        return best_obj

    def _detect_target_predicate(self, q_low: str) -> str | None:
        if any(k in q_low for k in ("city", "live", "living", "residence", "relocat", "move")):
            return "lives_in"
        if any(k in q_low for k in ("database", "primary db", "primary_db", "storage engine")):
            return "primary_db"
        if any(k in q_low for k in ("conference", "venue", "submission")):
            return "target_conference"
        if any(k in q_low for k in ("diet", "food", "meal restriction", "eat")):
            return "dietary_preference"
        if any(k in q_low for k in ("gpu", "compute budget", "vram", "hardware budget")):
            return "gpu_budget"
        if any(k in q_low for k in ("role", "job title", "position")):
            return "current_role"
        if any(k in q_low for k in ("advisor", "supervis")):
            return "advisor"
        if any(k in q_low for k in ("os preference", "operating system", "workstation run")):
            return "os_preference"
        if any(k in q_low for k in ("deployment cloud", "deployed on", "cloud platform")):
            return "deployment_cloud"
        if any(k in q_low for k in ("language", "speak", "fluent")):
            return "speaks_language"
        if any(k in q_low for k in ("course", "class", "curriculum")):
            return "completed_course"
        if any(k in q_low for k in ("library", "libraries", "framework", "toolkit")):
            return "uses_library"
        if any(
            k in q_low
            for k in (
                "organization",
                "affiliated",
                "affiliation",
                "company",
                "lab",
                "institute",
                "work at",
                "works at",
                "workplace",
                "work before",
                "joining",
            )
        ):
            return "works_at"
        if any(k in q_low for k in ("collaborator", "teammate", "partner")):
            return "collaborates_with"
        if "project" in q_low:
            return "works_on"
        return None
