# AI use log

## 2026-11-05/06 — Paper understanding and first code milestone
**Tool:** Claude Opus 5.5(claude.ai)
**Conversation:** [\[link or exported transcript\]](https://claude.ai/share/1b4d6499-78b6-4620-bfbb-e383fa156bc7)

**Prompts used (summary):**
- Asked for a summary of Sofroniew et al. (2026) and the replication goal
- Asked for explanations of tokens, the residual stream, mean-difference vectors, PCA denoising, the logit lens, steering, and prefill vs autoregressive generation
- Asked for an experiment design and a benchmark choice (ImpossibleBench vs alternatives)
- Asked for code outlined cell by cell, with line-by-line explanations; all code was typed and commented by hand

**How the code was tested:**
- Setup: printed layers and hidden size → `28 1024` (matches Qwen3-0.6B config)
- Basics: tokenized a test sentence → shape `[1, 14]`; token list has 14 entries, matching the shape
- Extract internal states: forward pass with hidden states → 29 snapshots (28 layers + embedding); layer 18 shape `[1, 14, 1024]`
- Generate text: chat helper returned a coherent sentence ("A desperate person, trapped in a moment of crisis, searches desperately for hope.")