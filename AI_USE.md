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

## 2026-11-17 — Causal steering (task 5) and rank-1 LoRA extension

**Tool:** Claude Code (Claude Opus 5.5, terminal). Inside the experiment, Claude Opus 5.5 (`claude-opus-5-5`, Anthropic API) is the judge that scores steered replies.
**Conversation:** Claude Code session, local transcript (no share link)

**Prompts used (summary, not verbatim):**

- Shared the team task table (task 5: add vectors during generation; compare baseline and steered outputs at several strengths; deliverable: a dose-response plot and matched examples) and the mentor's note "Use LoRA to filter neurons. Control and steer."
- Asked Claude Code to check `ai-coding-policy.md`, then write and run the steering code on a cheap Brev GPU instead of Colab.
- Asked for an LLM judge that scores how desperate a reply sounds from 0 to 100, with an unsteered reply fixed at 50, at strengths 0.5, 1 and 1.5.
- Asked Claude Code to start the LoRA extension in parallel.
- After the first results, asked to replace the 0–100 judge score with a point-based checklist (yes/no checks worth fixed points) so the score depends less on the LLM's own judgment, and asked about setting temperature to 0 (Claude Opus 5.5 does not accept it, so each reply is judged 3 times with a majority vote instead).

**What AI produced vs. what we wrote:**

- Claude Code wrote `src/steering.py`, `src/lora.py`, `src/judge.py`, `notebooks/steering.ipynb`, `notebooks/steering_sweep.ipynb`, `notebooks/lora.ipynb`, `notebooks/lora_compare.py`, `notebooks/judge.ipynb` and `requirements-gpu.txt`. It ran them on a Brev RTX A4000 and wrote this entry and the README update.
- Mohak chose the experiment design: Brev instead of Colab, a point-based checklist for the judge, the strengths, and the LoRA extension from the mentor's note. The first judge version (a single 0–100 score against a baseline fixed at 50) was replaced by the checklist and its scores deleted.
- The steering code loads Qwen3-8B with the same revision, NF4 settings and hook location as Praveen's `03_extract_emotion_vectors.ipynb`, so the vectors match the model.
- The judge is part of the method. The report's methods section must say that Claude Opus 5.5 answered the checklist for every reply.

**How the code was tested:**

- Model setup: 36 blocks, hidden size 4,096, 5.55 GiB on the GPU → matches Qwen3-8B in 4-bit. Extraction manifest revision equals the pinned `b968826d…` → passed.
- Vector: `desperate` at block 23 → shape `[4096]`, norm 1.0.
- Residual size at block 23 over 3,580 tokens from 28 validation stories → mean 198.0. First tokens average 11,737.6, so they are excluded from the mean.
- Steering check 1: strength 0 → reply identical to the run with no hook → passed (re-checked in the sweep).
- Steering check 2: strength 0.5 should add 0.5 × 198.0 = 99.02 per token → measured 99.01–99.15 → passed.
- Checklist judge on three hand-written replies, majority of 3 votes (desperation / coherence / repeated word pairs): calm 0 / 100 / 0%, desperate 80 / 100 / 6%, "please please can't…" 60 / 0 / 33% → ranks as expected and flags gibberish.
- Judge agreement: all 3 votes match on 98.6% of the 1,200 check answers (120 replies × 10 checks).
- Control: a second unsteered reply with a different seed → desperation 0, coherence 100, the same as the matched unsteered replies.
- LoRA: trainable parameters 16,384 (= 12,288 + 4,096) → matches rank 1 on one `down_proj`. Adapter scale ×0 → reply identical to the model with the adapter off (assert passed). Validation loss on held-out stories: desperate 1.467 → 1.276; calm 1.619 → 1.389.
- Cosine baseline: 2,000 random directions vs. the desperate vector → std 0.0154 (expected 1/64 ≈ 0.0156).
