# AI use log

## 2026-11-05/06 — Paper understanding and first code milestone

**Tool:** Claude Opus 5.5(claude.ai)
**Conversation:** [[link or exported transcript]](https://claude.ai/share/1b4d6499-78b6-4620-bfbb-e383fa156bc7)

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

## 2026-11-06 — Pilot story generation and Ollama setup

**Tool:** Codex (OpenAI desktop app)

**Prompts used (summary, not verbatim):**

- Asked for a plan for generating stories for each emotion–topic combination using Qwen through Ollama.
- Asked for Python code to generate a 56-story pilot with Qwen3-4B: fourteen emotions, two topics, and two stories per combination, using the story prompt adapted from Sofroniew et al.
- Asked for explanations of the notebook sections and individual lines of code.
- Supplied a DataHub connection-refused traceback and asked what caused the error and how to fix it.

**How the code was tested:**

- Model setup: inspected the model information and server logs; Ollama detected the GTX 1080 Ti using its CUDA 12 backend.
- Prompt: inspected the rendered emotion, topic, story count, and indirect-expression instructions before generation.
- Parsing and storage: inspected returned text and saved story records for correct separation and removal of formatting markers.
- Dataset structure: checked the pilot’s story counts and intended labels against the 14 × 2 × 2 design.
- Content: manually inspected stories for emotional clarity, premise adherence, explicit emotion language, and repeated phrasing.

## 2026-11-17: Validation and generalization (Member 2, Flavia): Claude Code

**Tool:** Claude Code (Claude Opus 5.5) in the VS Code extension, working in this repo on branch `flavia/validation-generalization`.

**Prompts used (summary):** Flavia gave a written spec. It covered reading notebook 03 and copying its conventions, then implementing Part A (held-out stories, a label-permutation null and a single test evaluation) and Parts B/C (implicit and numerical prompt probes for Colab). Flavia fixed the method (score formulas, splits, null design, readout position, prompt format). She asked for draft prompts to review, a CPU dry run, tests before any results, and no fabricated numbers.

**What Claude Code wrote (all of it, not typed by hand):**

- `.gitignore`: outputs/, *.npz, output_drive/, __pycache__/, .ipynb_checkpoints/, .venv/.
- `validation_generalization/A_heldout_stories.ipynb`: all code and markdown. This covers loading and checks, unit checks, metrics, the permutation null, figures, the test evaluation and the summary file.
  - Implementation details Claude chose: the best-block tie-break rule, the tie-aware vectorized AUROC for the null loop, `test_lock.json`, population SD for the z-scores, and the plot styling.
- `validation_generalization/BC_prompt_probes_colab.ipynb`: all code and markdown.
  - The install and model-loading cells are copied from notebook 03.
  - Claude also wrote the forbidden-term checker, prompt building, the readout hook, the hook tests, the neutral baseline, the B and C scoring, plots, the run record, and the DRY_RUN mode (Qwen3-0.6B on CPU with random vectors).
- `validation_generalization/prompts.json`: **draft** prompts for Flavia to review.
  - 35 Part B scenarios, 20 neutral prompts, and the X values for the 6 Part C templates. Flavia specified which C templates to include.
  - Extra forbidden terms beyond the dataset manifest.
  - The Tylenol, missing-dog and runway wording is adapted from the paper, not quoted verbatim.
- `validation_generalization/README.md` and this entry.

**How the code was tested (all run locally on 2026-11-17; details in `validation_generalization/README.md`):**

- **Archive checks:** keys, shapes, label order and layer order are read from the files. Story order matches `stories.jsonl`. Split sizes and topics per split are as designed.
- **Vector rebuild:** vectors rebuilt from the training rows with notebook 03's recipe match the saved `emotion_means`, `center`, `raw_vectors` and `unit_vectors` (max abs diff 0).
- **Scoring checks:**
  - A synthetic orientation test passes.
  - Score shape and a spot check against an explicit dot product pass.
  - The fast AUROC equals sklearn, both on a synthetic case with ties and on all real validation columns.
  - The null code path with unshuffled labels reproduces the real metrics.
- **Training sanity (in-sample):** top-1 is 0.463 at block 23, well above chance (0.071).
- **Bug caught by tests:** the first null run tripped an assertion on tied scores, so the AUROC was rewritten to use average ranks.
- **B/C dry run** (Qwen3-0.6B on CPU, random vectors; code check only):
  - The forbidden-term check passes on all 115 texts.
  - The chat-template ending and last token are verified.
  - The hook returns shape (1024,). Logits are unchanged with the hook attached and after removal (max diff 0.0). The hooked state equals `hidden_states[24]`.
  - Passes under transformers 4.57.6 and 5.20.0.
  - A failed hook-removal assertion under 5.20.0 showed that the library adds its own capture hook. The test now checks for our own hook id instead.
- **Not run:** Parts B/C on Qwen3-8B, which needs a CUDA GPU (Colab).

**Human review still needed:** Flavia reviews `prompts.json`, approves which figures get committed, and checks the Part A interpretation.
