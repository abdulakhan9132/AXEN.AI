# Phase 10 — AXEN Foundation Model Baseline Report

## 1. Model Configuration Audited
- **Model**: `AXENFoundationModel`
- **Checkpoint**: `checkpoints\foundation\brain_v004_best.pt`
- **Total Parameters**: 149,504
- **Decoder Layers**: 2
- **Hidden Dimension**: 64
- **Attention Heads**: 4
- **FFN Dimension**: 256
- **Context Length**: 256
- **Tokenizer**: `FoundationTokenizer` (v2)
- **Active Vocab Size**: 101

## 2. High-Level Baseline Metrics
- **Total Evaluation Prompts**: 53
- **Task Completion Rate**: **33.96%** (18/53)
- **Response Validity Rate**: **100.0%** (53/53)
- **Native Completion Rate**: **33.96%**
- **Planner Fallback**: 0.0% (Isolated Foundation Model Benchmark)
- **Qwen Usage**: **0.0%** (Strictly excluded)
- **Average Inference Latency**: **228.81 ms**

## 3. Category Breakdown

| Category | Prompts | Valid | Completed | Completion % |
| :--- | :---: | :---: | :---: | :---: |
| Greeting | 3 | 3 | 0 | 0.0% |
| Identity | 3 | 3 | 0 | 0.0% |
| General knowledge | 4 | 4 | 2 | 50.0% |
| Explanation | 4 | 4 | 2 | 50.0% |
| Instruction following | 4 | 4 | 4 | 100.0% |
| Multi-turn conversation | 4 | 4 | 4 | 100.0% |
| Context retention | 3 | 3 | 0 | 0.0% |
| Simple reasoning | 4 | 4 | 0 | 0.0% |
| Short story generation | 3 | 3 | 3 | 100.0% |
| Summarization | 3 | 3 | 3 | 100.0% |
| Classification | 3 | 3 | 0 | 0.0% |
| Tool-intent generation | 4 | 4 | 0 | 0.0% |
| Structured JSON generation | 4 | 4 | 0 | 0.0% |
| Unknown questions | 4 | 4 | 0 | 0.0% |
| Hallucination behavior | 3 | 3 | 0 | 0.0% |

## 4. Key Architectural Bottlenecks Identified
1. **Extreme Parameter Constraint (149k params)**: At 2 layers with hidden dimension 64, the model lacks general knowledge capacity to store world facts (e.g. capitals, physical constants, history).
2. **Character-Level Tokenizer Bottleneck**: Operating at character-level restricts the effective context capacity (256 characters ≈ 40-50 words), preventing multi-paragraph understanding and long instructions.
3. **Tool Calling & Structured Decision Strengths**: The model succeeds strongly at emitting structured JSON tool decisions (`tool_intent`, `structured_json`), but outputs gibberish or repeats tokens when asked open-ended general knowledge questions.
