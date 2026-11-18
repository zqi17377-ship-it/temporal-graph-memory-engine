# Long-Horizon Cross-Session Memory Benchmark Report

## 1. Dataset & Experimental Protocol
- **Personas**: 4 longitudinal CS/AI engineering users (`Alex Chen`, `Su Yan`, `Zhao Kai`, `Lin Wei`)
- **Sessions & Horizon**: 20 sessions, **240 dialogue turns** (~68.3% casual chitchat/filler noise)
- **Explicit State Overrides & Retractions**: 21 cross-session updates
- **Evaluation Probes**: **60 QA probes** (15 per category across 4 categories)

## 2. Main Ablation: 4 Memory Architectures

| Memory System | Overall Acc | Long-Range (15) | Conflict Override (15) | Historical Lineage (15) | Multi-Hop Graph (15) | Conflict Precision | Avg Tokens | Token Reduction vs Full | P95 Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sliding Window (Last 10 Turns)** | **5.0%** | 0.0% | 6.7% | 0.0% | 13.3% | 100.0% | 179.6 | +82.8% | 0.02 |
| **Full History Context Stuffing** | **61.7%** | 93.3% | 6.7% | 53.3% | 93.3% | 6.7% | 1041.1 | +0.0% | 0.10 |
| **Flat Vector RAG Memory (Top-5 Chunks)** | **51.7%** | 93.3% | 13.3% | 53.3% | 46.7% | 15.4% | 94.6 | +90.9% | 5.95 |
| **Ours: Hierarchical Temporal Graph + Vector (theta=0.25)** | **96.7%** | 93.3% | 100.0% | 93.3% | 100.0% | 100.0% | 147.3 | +85.8% | 7.95 |

## 3. Ebbinghaus Forgetting Threshold Sweep (Compression vs Accuracy Pareto Frontier)

| Consolidation Threshold $\theta$ | Active Edges Retained | Edges Archived | Graph Compression Rate | Overall Acc | Multi-Hop Acc | Long-Range Acc | Conflict Override Acc | Avg Prompt Tokens |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `$\theta = 0.00$` | 159 | 0 | **0.0%** | **96.7%** | 100.0% | 93.3% | 100.0% | 148.0 |
| `$\theta = 0.25$` | 61 | 98 | **61.6%** | **96.7%** | 100.0% | 93.3% | 100.0% | 147.3 |
| `$\theta = 0.55$` | 18 | 141 | **88.7%** | **80.0%** | 73.3% | 93.3% | 60.0% | 75.5 |

## 4. Failure Case Attribution Across Architectures

- **Sliding Window (Last 10 Turns)**: `missing_recall_evicted_or_unretrieved`: 56, `incomplete_historical_lineage`: 1
- **Full History Context Stuffing**: `stale_conflict_hallucination`: 15, `incomplete_historical_lineage`: 7, `wrong_or_incomplete_entity`: 1
- **Flat Vector RAG Memory (Top-5 Chunks)**: `stale_conflict_hallucination`: 12, `missing_recall_evicted_or_unretrieved`: 9, `incomplete_historical_lineage`: 6, `broken_multi_hop_chain`: 1, `wrong_or_incomplete_entity`: 1
- **Ours: Hierarchical Temporal Graph + Vector (theta=0.25)**: `wrong_or_incomplete_entity`: 1, `incomplete_historical_lineage`: 1
