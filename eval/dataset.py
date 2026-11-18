"""Long-Horizon Cross-Session Benchmark Dataset (4 Personas, 20 Sessions, 240 Turns, 60 QA Probes)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from graph_memory.models import DialogueTurn

ProbeCategory = Literal[
    "single_hop_long_range",
    "fact_conflict_override",
    "temporal_historical_lineage",
    "multi_hop_graph_reasoning",
]


@dataclass(frozen=True)
class QAProbe:
    probe_id: str
    user_id: str
    user_display_name: str
    category: ProbeCategory
    question: str
    expected_keywords: list[str]
    forbidden_stale: list[str] = field(default_factory=list)
    query_turn: int = 60


@dataclass(frozen=True)
class PersonaHistory:
    user_id: str
    display_name: str
    turns: list[DialogueTurn]
    probes: list[QAProbe]


_FILLER_TEMPLATES: list[str] = [
    "Had spicy ramen for lunch today before heading back to the lab.",
    "Complaining about rainy campus weather all morning, forgot my umbrella again.",
    "Grabbed an iced oat latte from the corner cafe to survive afternoon code review.",
    "Browsing threads on mechanical keyboard switches during my break.",
    "Haha morning standup was boring today, mostly formatting nitpicks.",
    "Had beef pho with classmates and talked about the semester schedule.",
    "Reading blog posts on gaming monitors while waiting for CI builds to finish.",
    "Grabbed a cold brew and fixed an annoying off-by-one unit test.",
    "Chatting about subway delays on Line 4 this morning, traffic was wild.",
    "Ate xiao long bao near the south gate, super crowded today.",
    "Complaining about cafeteria queues at noon, took twenty minutes.",
    "Had a matcha latte after debugging a flaky async timeout.",
    "Grabbed a sesame bagel before the morning seminar started.",
    "Browsing threads on ergonomic office chairs for my dorm desk.",
    "Had a hot americano while reviewing pull request comments.",
    "Chatting about weekend sci-fi movies with my labmates.",
    "Ate roast duck rice at the student center second floor.",
    "Reading blog posts on campus stray cats near the library.",
    "Grabbed an iced lemon tea on the walk back from the gym.",
    "Browsing threads on indie game soundtracks while compiling.",
    "Had a berry smoothie after an intense debugging session.",
    "Complaining about laundry room queues in the graduate dorm.",
    "Ate pork dumplings for a quick dinner before evening experiments.",
    "Chatting about badminton court booking rules this week.",
    "Grabbed a tuna sandwich between two back-to-back meetings.",
    "Reading blog posts on desk lamp lighting for late-night coding.",
    "Had a chicken wrap before afternoon office hours.",
    "Haha git rebase had three tiny merge conflicts in config files.",
    "Stretched my shoulders after staring at terminal logs for two hours.",
    "Cleaned up unused scratch files on my laptop desktop today.",
    "Forgot my charger in the seminar room and walked back to fetch it.",
    "Watched snow flurries outside the lab window for five minutes.",
    "Organized my browser bookmarks folder during lunch break.",
    "Swapped out worn keycaps on my travel keyboard.",
    "Took a quick power nap before the afternoon reading group.",
    "Helped a freshman find the systems lab classroom.",
    "Refilled my water bottle twice during the long lecture.",
    "Wiped down my whiteboard before sketching the new diagram.",
    "Checked campus shuttle bus departure times on my phone.",
    "Tuned terminal font ligatures for better readability.",
    "Walked around the track for twenty minutes after dinner.",
    "Updated colour theme in my editor to reduce eye strain.",
    "Sorted PDF papers into folders by reading priority.",
    "Haha the vending machine ran out of sparkling water again.",
]


def _build_session_turns(
    user_id: str,
    scripted_by_turn: dict[int, str],
    filler_offset: int = 0,
) -> list[DialogueTurn]:
    """Construct 60 dialogue turns across 5 sessions (12 turns/session), padding with realistic noise."""
    turns: list[DialogueTurn] = []
    filler_idx = filler_offset
    for t_id in range(1, 61):
        session_num = (t_id - 1) // 12 + 1
        session_id = f"{user_id}_session_{session_num:02d}"
        if t_id in scripted_by_turn:
            content = scripted_by_turn[t_id]
        else:
            content = _FILLER_TEMPLATES[filler_idx % len(_FILLER_TEMPLATES)]
            filler_idx += 1
        timestamp = f"2026-0{session_num}-{(t_id % 27) + 1:02d}T10:{t_id % 60:02d}:00Z"
        turns.append(
            DialogueTurn(
                turn_id=t_id,
                session_id=session_id,
                user_id=user_id,
                role="user",
                content=content,
                timestamp=timestamp,
            )
        )
    return turns


def build_benchmark_dataset() -> list[PersonaHistory]:
    """Build 4 longitudinal personas (240 turns across 20 sessions) + 60 balanced QA probes."""

    # =========================================================================
    # Persona 1: Alex Chen (Distributed Graph & Systems CS Junior)
    # =========================================================================
    alex_script = {
        # Session 1 (Turns 1..12) — Long-range facts >= 51 turns before Turn 60!
        1: "My academic advisor is Prof. Hao Zhang.",
        2: "I completed course CS224N and I completed course MIT 6.824.",
        3: "My OS preference is Arch Linux.",
        4: "I speak Mandarin and I am also fluent in Japanese.",
        5: "Currently I live in Beijing.",
        6: "My primary database is MySQL.",
        7: "I started leading Project Helios.",
        8: "I actively use PyTorch and I actively use LangChain.",
        # Session 2 (Turns 13..24)
        14: "My GPU budget is Single RTX 3090 (24GB).",
        17: "I relocated to Hangzhou.",
        19: "For Project Helios, I teamed up with Dr. Mei Zhou.",
        21: "We switched our primary database from MySQL to MongoDB.",
        # Session 3 (Turns 25..36)
        27: "Our GPU budget was upgraded to Dual RTX 4090 (48GB).",
        30: "Project Helios uses primary database ClickHouse.",
        33: "I stopped using LangChain.",
        # Session 4 (Turns 37..48)
        39: "I settled down in Shanghai.",
        42: "Dr. Mei Zhou is affiliated with Tsinghua AIR.",
        44: "Dr. Mei Zhou targets conference NeurIPS.",
        46: "We migrated our primary database from MongoDB to PostgreSQL.",
        # Session 5 (Turns 49..60)
        50: "Project Helios is deployed on AWS EKS.",
        53: "Compute budget is now 4x A100 (320GB).",
    }

    alex_probes = [
        # single_hop_long_range (4)
        QAProbe("p1_lr_1", "user_chen", "Alex Chen", "single_hop_long_range",
                "Who is Alex Chen's academic advisor?", ["Prof. Hao Zhang"]),
        QAProbe("p1_lr_2", "user_chen", "Alex Chen", "single_hop_long_range",
                "Which courses has Alex Chen completed?", ["CS224N", "MIT 6.824"]),
        QAProbe("p1_lr_3", "user_chen", "Alex Chen", "single_hop_long_range",
                "What is Alex Chen's OS preference?", ["Arch Linux"]),
        QAProbe("p1_lr_4", "user_chen", "Alex Chen", "single_hop_long_range",
                "Which natural languages does Alex Chen speak?", ["Mandarin", "Japanese"]),
        # fact_conflict_override (4)
        QAProbe("p1_co_1", "user_chen", "Alex Chen", "fact_conflict_override",
                "Which city does Alex Chen currently live in?",
                ["Shanghai"], forbidden_stale=["Beijing", "Hangzhou"]),
        QAProbe("p1_co_2", "user_chen", "Alex Chen", "fact_conflict_override",
                "What is Alex Chen's primary database right now?",
                ["PostgreSQL"], forbidden_stale=["MySQL", "MongoDB"]),
        QAProbe("p1_co_3", "user_chen", "Alex Chen", "fact_conflict_override",
                "Which libraries does Alex Chen actively use now?",
                ["PyTorch"], forbidden_stale=["LangChain"]),
        QAProbe("p1_co_4", "user_chen", "Alex Chen", "fact_conflict_override",
                "What is Alex Chen's current GPU budget?",
                ["4x A100 (320GB)"], forbidden_stale=["Single RTX 3090", "Dual RTX 4090"]),
        # temporal_historical_lineage (4)
        QAProbe("p1_th_1", "user_chen", "Alex Chen", "temporal_historical_lineage",
                "Which city did Alex Chen live in before moving to Hangzhou?", ["Beijing"]),
        QAProbe("p1_th_2", "user_chen", "Alex Chen", "temporal_historical_lineage",
                "How did Alex Chen's city of residence evolve across sessions?",
                ["Beijing", "Hangzhou", "Shanghai"]),
        QAProbe("p1_th_3", "user_chen", "Alex Chen", "temporal_historical_lineage",
                "What was Alex Chen's primary database before switching to PostgreSQL?", ["MongoDB"]),
        QAProbe("p1_th_4", "user_chen", "Alex Chen", "temporal_historical_lineage",
                "What was Alex Chen's initial GPU budget originally?", ["Single RTX 3090 (24GB)"]),
        # multi_hop_graph_reasoning (4)
        QAProbe("p1_mh_1", "user_chen", "Alex Chen", "multi_hop_graph_reasoning",
                "Which organization is the collaborator on the project led by Alex Chen affiliated with?",
                ["Tsinghua AIR"]),
        QAProbe("p1_mh_2", "user_chen", "Alex Chen", "multi_hop_graph_reasoning",
                "Which conference does the collaborator on Alex Chen's project target?",
                ["NeurIPS"]),
        QAProbe("p1_mh_3", "user_chen", "Alex Chen", "multi_hop_graph_reasoning",
                "Which primary database does the project led by Alex Chen rely on?",
                ["ClickHouse"]),
        QAProbe("p1_mh_4", "user_chen", "Alex Chen", "multi_hop_graph_reasoning",
                "Which cloud platform is the project led by Alex Chen deployed on?",
                ["AWS EKS"]),
    ]

    # =========================================================================
    # Persona 2: Su Yan (Multimodal Vision & Embodied AI Intern)
    # =========================================================================
    su_script = {
        # Session 1 (Turns 1..12) — Long-range facts >= 51 turns before Turn 60!
        1: "My academic advisor is Prof. Qian Wu.",
        2: "I completed course CS231N and I completed course CS336.",
        3: "I speak English and I am also fluent in French.",
        4: "My OS preference is Ubuntu 24.04.",
        5: "Currently I work at SenseTime.",
        6: "My target conference is CVPR.",
        7: "My dietary preference is Vegetarian.",
        8: "I started leading Project Prism.",
        # Session 2 (Turns 13..24)
        15: "I joined ByteDance.",
        18: "For Project Prism, I teamed up with Dr. Wei Lin.",
        20: "We switched our target conference from CVPR to ICML.",
        22: "Project Prism is deployed on GCP Vertex.",
        # Session 3 (Turns 25..36)
        28: "Dr. Wei Lin is affiliated with Zhejiang Lab.",
        31: "Dr. Wei Lin uses library Triton.",
        34: "I switched to a Vegan diet.",
        # Session 4 (Turns 37..48)
        40: "I transferred to Shanghai AI Lab.",
        43: "We retargeted our submission from ICML to ICLR.",
        45: "Project Prism uses primary database Qdrant.",
        # Session 5 (Turns 49..60)
        52: "Dr. Wei Lin lives in Hangzhou.",
    }

    su_probes = [
        # single_hop_long_range (4)
        QAProbe("p2_lr_1", "user_su", "Su Yan", "single_hop_long_range",
                "Who is Su Yan's academic advisor?", ["Prof. Qian Wu"]),
        QAProbe("p2_lr_2", "user_su", "Su Yan", "single_hop_long_range",
                "Which courses has Su Yan completed?", ["CS231N", "CS336"]),
        QAProbe("p2_lr_3", "user_su", "Su Yan", "single_hop_long_range",
                "Which languages does Su Yan speak?", ["English", "French"]),
        QAProbe("p2_lr_4", "user_su", "Su Yan", "single_hop_long_range",
                "What is Su Yan's OS preference?", ["Ubuntu 24.04"]),
        # fact_conflict_override (3)
        QAProbe("p2_co_1", "user_su", "Su Yan", "fact_conflict_override",
                "Which organization does Su Yan currently work at?",
                ["Shanghai AI Lab"], forbidden_stale=["SenseTime", "ByteDance"]),
        QAProbe("p2_co_2", "user_su", "Su Yan", "fact_conflict_override",
                "What is Su Yan's current target conference?",
                ["ICLR"], forbidden_stale=["CVPR", "ICML"]),
        QAProbe("p2_co_3", "user_su", "Su Yan", "fact_conflict_override",
                "What is Su Yan's current dietary preference?",
                ["Vegan"], forbidden_stale=["Vegetarian"]),
        # temporal_historical_lineage (4)
        QAProbe("p2_th_1", "user_su", "Su Yan", "temporal_historical_lineage",
                "Where did Su Yan work before joining ByteDance?", ["SenseTime"]),
        QAProbe("p2_th_2", "user_su", "Su Yan", "temporal_historical_lineage",
                "How did Su Yan's workplace affiliation evolve over time?",
                ["SenseTime", "ByteDance", "Shanghai AI Lab"]),
        QAProbe("p2_th_3", "user_su", "Su Yan", "temporal_historical_lineage",
                "Which conference was Su Yan targeting before retargeting to ICLR?", ["ICML"]),
        QAProbe("p2_th_4", "user_su", "Su Yan", "temporal_historical_lineage",
                "What was Su Yan's dietary preference originally?", ["Vegetarian"]),
        # multi_hop_graph_reasoning (4)
        QAProbe("p2_mh_1", "user_su", "Su Yan", "multi_hop_graph_reasoning",
                "Which organization is the collaborator on the project led by Su Yan affiliated with?",
                ["Zhejiang Lab"]),
        QAProbe("p2_mh_2", "user_su", "Su Yan", "multi_hop_graph_reasoning",
                "Which library does the collaborator on Su Yan's project use?",
                ["Triton"]),
        QAProbe("p2_mh_3", "user_su", "Su Yan", "multi_hop_graph_reasoning",
                "Which primary database does the project led by Su Yan rely on?",
                ["Qdrant"]),
        QAProbe("p2_mh_4", "user_su", "Su Yan", "multi_hop_graph_reasoning",
                "Which city does the collaborator on Su Yan's project live in?",
                ["Hangzhou"]),
    ]

    # =========================================================================
    # Persona 3: Zhao Kai (Storage Engines & Rust Infrastructure Engineer)
    # =========================================================================
    zhao_script = {
        # Session 1 (Turns 1..12) — Long-range facts >= 51 turns before Turn 60!
        1: "My academic advisor is Prof. Bohan Liu.",
        2: "I completed course CMU 15-445 and I completed course CMU 15-721.",
        3: "My OS preference is NixOS.",
        4: "My deployment cloud is Aliyun ACK.",
        5: "Currently I live in Nanjing.",
        6: "My current role is Backend Intern.",
        7: "I started leading Project Kronos.",
        8: "Project Kronos uses primary database SQLite.",
        # Session 2 (Turns 13..24)
        16: "I was promoted to Junior Systems Engineer.",
        19: "For Project Kronos, I teamed up with Dr. Yue Tang.",
        22: "I moved down to Shenzhen.",
        # Session 3 (Turns 25..36)
        27: "Dr. Yue Tang is affiliated with DeepSeek.",
        30: "Project Kronos migrated primary database from SQLite to DuckDB.",
        34: "My primary database is Redis.",
        # Session 4 (Turns 37..48)
        39: "I transitioned to role Distributed Storage Lead.",
        42: "Dr. Yue Tang targets conference OSDI.",
        45: "We switched our primary database from Redis to PostgreSQL.",
        # Session 5 (Turns 49..60)
        51: "Dr. Yue Tang uses library Ray.",
    }

    zhao_probes = [
        # single_hop_long_range (4)
        QAProbe("p3_lr_1", "user_zhao", "Zhao Kai", "single_hop_long_range",
                "Who is Zhao Kai's academic advisor?", ["Prof. Bohan Liu"]),
        QAProbe("p3_lr_2", "user_zhao", "Zhao Kai", "single_hop_long_range",
                "Which database courses has Zhao Kai completed?", ["CMU 15-445", "CMU 15-721"]),
        QAProbe("p3_lr_3", "user_zhao", "Zhao Kai", "single_hop_long_range",
                "What is Zhao Kai's OS preference?", ["NixOS"]),
        QAProbe("p3_lr_4", "user_zhao", "Zhao Kai", "single_hop_long_range",
                "What is Zhao Kai's deployment cloud?", ["Aliyun ACK"]),
        # fact_conflict_override (4)
        QAProbe("p3_co_1", "user_zhao", "Zhao Kai", "fact_conflict_override",
                "Which city does Zhao Kai currently live in?",
                ["Shenzhen"], forbidden_stale=["Nanjing"]),
        QAProbe("p3_co_2", "user_zhao", "Zhao Kai", "fact_conflict_override",
                "What is Zhao Kai's current role?",
                ["Distributed Storage Lead"],
                forbidden_stale=["Backend Intern", "Junior Systems Engineer"]),
        QAProbe("p3_co_3", "user_zhao", "Zhao Kai", "fact_conflict_override",
                "What is Zhao Kai's personal primary database right now?",
                ["PostgreSQL"], forbidden_stale=["Redis"]),
        QAProbe("p3_co_4", "user_zhao", "Zhao Kai", "fact_conflict_override",
                "Which primary database does Project Kronos currently use?",
                ["DuckDB"], forbidden_stale=["SQLite"]),
        # temporal_historical_lineage (4)
        QAProbe("p3_th_1", "user_zhao", "Zhao Kai", "temporal_historical_lineage",
                "Which city did Zhao Kai live in before moving to Shenzhen?", ["Nanjing"]),
        QAProbe("p3_th_2", "user_zhao", "Zhao Kai", "temporal_historical_lineage",
                "How did Zhao Kai's job role evolve across sessions?",
                ["Backend Intern", "Junior Systems Engineer", "Distributed Storage Lead"]),
        QAProbe("p3_th_3", "user_zhao", "Zhao Kai", "temporal_historical_lineage",
                "Which primary database did Project Kronos initially use?", ["SQLite"]),
        QAProbe("p3_th_4", "user_zhao", "Zhao Kai", "temporal_historical_lineage",
                "What was Zhao Kai's role before Distributed Storage Lead?",
                ["Junior Systems Engineer"]),
        # multi_hop_graph_reasoning (4)
        QAProbe("p3_mh_1", "user_zhao", "Zhao Kai", "multi_hop_graph_reasoning",
                "Which organization is the collaborator on the project led by Zhao Kai affiliated with?",
                ["DeepSeek"]),
        QAProbe("p3_mh_2", "user_zhao", "Zhao Kai", "multi_hop_graph_reasoning",
                "Which conference does the collaborator on Zhao Kai's project target?",
                ["OSDI"]),
        QAProbe("p3_mh_3", "user_zhao", "Zhao Kai", "multi_hop_graph_reasoning",
                "Which primary database does the project led by Zhao Kai rely on?",
                ["DuckDB"], forbidden_stale=["SQLite"]),
        QAProbe("p3_mh_4", "user_zhao", "Zhao Kai", "multi_hop_graph_reasoning",
                "Which library does the collaborator on Zhao Kai's project use?",
                ["Ray"]),
    ]

    # =========================================================================
    # Persona 4: Lin Wei (NLP Reasoning & LLM Inference Researcher)
    # =========================================================================
    lin_script = {
        # Session 1 (Turns 1..12) — Long-range facts >= 51 turns before Turn 60!
        1: "My academic advisor is Prof. Ming Xu.",
        2: "I completed course CS285 and I completed course Stanford CS229.",
        3: "I speak Cantonese and I am also fluent in German.",
        4: "My GPU budget is Dual RTX 4090 (48GB).",
        5: "My dietary preference is Halal.",
        6: "My target conference is ACL.",
        7: "I actively use vLLM and I actively use TensorFlow.",
        8: "I started leading Project Nebula.",
        # Session 2 (Turns 13..24)
        15: "For Project Nebula, I teamed up with Dr. Chen Gao.",
        18: "We switched our target conference from ACL to EMNLP.",
        21: "I stopped using TensorFlow.",
        # Session 3 (Turns 25..36)
        27: "Dr. Chen Gao is affiliated with MSRA.",
        31: "I switched to a Lactose-Free diet.",
        34: "Project Nebula is deployed on Modal Serverless.",
        # Session 4 (Turns 37..48)
        39: "Our GPU budget was upgraded to 8x H100 (640GB).",
        43: "We retargeted our submission from EMNLP to NeurIPS.",
        46: "Project Nebula uses primary database Milvus.",
        # Session 5 (Turns 49..60)
        52: "Dr. Chen Gao lives in Beijing.",
    }

    lin_probes = [
        # single_hop_long_range (3 -> total 4+4+4+3 = 15!)
        QAProbe("p4_lr_1", "user_lin", "Lin Wei", "single_hop_long_range",
                "Who is Lin Wei's academic advisor?", ["Prof. Ming Xu"]),
        QAProbe("p4_lr_2", "user_lin", "Lin Wei", "single_hop_long_range",
                "Which courses has Lin Wei completed?", ["CS285", "Stanford CS229"]),
        QAProbe("p4_lr_3", "user_lin", "Lin Wei", "single_hop_long_range",
                "Which languages does Lin Wei speak?", ["Cantonese", "German"]),
        # fact_conflict_override (4 -> total 4+3+4+4 = 15!)
        QAProbe("p4_co_1", "user_lin", "Lin Wei", "fact_conflict_override",
                "What is Lin Wei's current GPU budget?",
                ["8x H100 (640GB)"], forbidden_stale=["Dual RTX 4090"]),
        QAProbe("p4_co_2", "user_lin", "Lin Wei", "fact_conflict_override",
                "What is Lin Wei's current dietary preference?",
                ["Lactose-Free"], forbidden_stale=["Halal"]),
        QAProbe("p4_co_3", "user_lin", "Lin Wei", "fact_conflict_override",
                "What is Lin Wei's current target conference?",
                ["NeurIPS"], forbidden_stale=["ACL", "EMNLP"]),
        QAProbe("p4_co_4", "user_lin", "Lin Wei", "fact_conflict_override",
                "Which libraries does Lin Wei actively use now?",
                ["vLLM"], forbidden_stale=["TensorFlow"]),
        # temporal_historical_lineage (3 -> total 4+4+4+3 = 15!)
        QAProbe("p4_th_1", "user_lin", "Lin Wei", "temporal_historical_lineage",
                "How did Lin Wei's target conference evolve across sessions?",
                ["ACL", "EMNLP", "NeurIPS"]),
        QAProbe("p4_th_2", "user_lin", "Lin Wei", "temporal_historical_lineage",
                "What was Lin Wei's GPU budget before upgrading to 8x H100 (640GB)?",
                ["Dual RTX 4090 (48GB)"]),
        QAProbe("p4_th_3", "user_lin", "Lin Wei", "temporal_historical_lineage",
                "What was Lin Wei's initial dietary preference?", ["Halal"]),
        # multi_hop_graph_reasoning (3 -> total 4+4+4+3 = 15!)
        QAProbe("p4_mh_1", "user_lin", "Lin Wei", "multi_hop_graph_reasoning",
                "Which organization is the collaborator on the project led by Lin Wei affiliated with?",
                ["MSRA"]),
        QAProbe("p4_mh_2", "user_lin", "Lin Wei", "multi_hop_graph_reasoning",
                "Which cloud platform is the project led by Lin Wei deployed on?",
                ["Modal Serverless"]),
        QAProbe("p4_mh_3", "user_lin", "Lin Wei", "multi_hop_graph_reasoning",
                "Which primary database does the project led by Lin Wei rely on?",
                ["Milvus"]),
    ]

    return [
        PersonaHistory(
            user_id="user_chen",
            display_name="Alex Chen",
            turns=_build_session_turns("user_chen", alex_script, filler_offset=0),
            probes=alex_probes,
        ),
        PersonaHistory(
            user_id="user_su",
            display_name="Su Yan",
            turns=_build_session_turns("user_su", su_script, filler_offset=3),
            probes=su_probes,
        ),
        PersonaHistory(
            user_id="user_zhao",
            display_name="Zhao Kai",
            turns=_build_session_turns("user_zhao", zhao_script, filler_offset=6),
            probes=zhao_probes,
        ),
        PersonaHistory(
            user_id="user_lin",
            display_name="Lin Wei",
            turns=_build_session_turns("user_lin", lin_script, filler_offset=9),
            probes=lin_probes,
        ),
    ]
