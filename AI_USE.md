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

## 2026-11-16 — Repo reorganization after code checkpoint feedback

**Tool:** Claude Code (Claude Opus 5.5, terminal)

**Prompts used (summary, not verbatim):**

- Shared the TA's checkpoint feedback (2/4: README missing setup, data, and run steps; no folder structure) and asked what was expected.
- Asked Claude Code to add a `.gitignore`, fill in the README from what is already in the repo, organize the code into folders, log the work here, and commit each step separately under my name.
- Asked what the difference between `src/` and `notebooks/` is and how to organize them.

**What AI produced vs. what we wrote:**

- Claude Code wrote `.gitignore` and this entry, created `results/`, and rewrote `README.md` using information from the old README, `requirements.txt`, and `emotion_vector_pipeline/README.md`.
- Claude Code moved `playground.ipynb` to `notebooks/` and moved our existing model-loading and `chat()` code into `src/model.py`. Our original comments were kept. The only logic change: `chat()` now takes `model` and `tok` as arguments instead of reading globals, and uses `model.device`.
- Claude Code did not change `emotion_vector_pipeline/` or `archive/`.

**How the changes were tested:**

- Notebook after the move: re-ran `notebooks/playground.ipynb` top to bottom from `notebooks/` → `28 1024`, `torch.Size([1, 14])`, `29`, `torch.Size([1, 14, 1024])`, and the same greedy-decoded sentence as on 2026-11-06 ("A desperate person, trapped in a moment of crisis, searches desperately for hope.").
- `.gitignore`: `git ls-files -ci --exclude-standard` → no tracked files ignored. `git check-ignore` → extracted JSONL files, `outputs/`, `.npz` files and the Drive archive are ignored; the committed dataset ZIP, `archive/data/` and `results/` are not.
- README data steps: ran the unzip and copy commands on a copy of the dataset ZIP → `data/emotion_stories_qwen_full1120.jsonl` has 1,120 lines, matching the dataset size.
