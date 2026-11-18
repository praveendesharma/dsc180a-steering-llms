# Validation and generalization of emotion vectors

Member 2 (Flavia) · DSC 180A · reproduces Sofroniew et al. (2026) §1.2 on Qwen3-8B, using Member 1's vectors from extraction run `ab1333f49024705c`.

| File | What it does | Where it runs |
| --- | --- | --- |
| `ABC_validation_walkthrough.ipynb` | **All three parts in one notebook.** It explains each step (what it does and why it matters) and ends each part with a result cell and a "what it means" section. The code is the same as in the two notebooks below. | Colab GPU: A, B and C for real. Laptop: Part A only, with B/C skipped; `BC_MODE=dryrun` gives the CPU code check. |
| `A_heldout_stories.ipynb` | **Part A.** Scores held-out stories with the emotion vectors at every block. Reports validation accuracy and AUROC, a label-permutation null, and a single test-split evaluation. | Local, CPU only; no model (about 40 s). |
| `BC_prompt_probes_colab.ipynb` | **Parts B & C.** Reads block-23 states for implicit-emotion scenarios (B) and numerical templates (C). | Colab with an NVIDIA GPU (NF4 needs CUDA). `BC_DRY_RUN=1` gives a CPU code check. |
| `prompts.json` | Part B scenarios, neutral prompts and Part C templates. **Draft, awaiting review.** | n/a |

## Inputs: where the archive is expected

Both notebooks read the unzipped Drive archive (`qwen3_8b_emotion_activations_vectors_ab1333f49024705c.zip`, about 1 GB; link in `emotion_vector_pipeline/README.md`) from one config variable at the top of each notebook:

```python
RUN_DIR = Path("../emotion_vector_pipeline/output_drive")   # relative to the notebook
```

By default that folder must contain `story_activations.npz` and `emotion_vectors.npz` directly (as well as `stories.jsonl` and `manifest.json`, which are used for cross-checks when present). **Anyone can change `RUN_DIR`** to point at their own copy, for example the pipeline README's `outputs/emotion_extraction/ab1333f49024705c/` or a Google Drive folder on Colab. `output_drive/` is in `.gitignore`, and the archive must never be committed.

Part B/C also reads the forbidden-term lists from `emotion_vector_pipeline/data/qwen3_8b_emotion_stories_1120_v1.zip` (committed).

## How to run

**Everything in one notebook:** open `ABC_validation_walkthrough.ipynb` and *Run all*.
- **On a Colab GPU runtime**, it runs Parts A, B and C for real (run the setup cell first and set `RUN_DIR`).
- **On a laptop**, it runs Part A and prints "skipped" for the B/C cells.
- **`BC_MODE=dryrun`** runs B/C as a CPU code check.

The separate notebooks below remain available.

**Part A (local):** use a Python environment with numpy, scikit-learn and matplotlib, then run all cells of `A_heldout_stories.ipynb`. Outputs go to `outputs/A/`.

**Part B/C (Colab):** open `BC_prompt_probes_colab.ipynb` on a GPU runtime, run the setup cell (mount Drive, clone this branch), set `RUN_DIR`, then run all cells. The install cell is the same as notebook 03's. Outputs go to `outputs/BC/`.

**Part B/C code check (CPU, local):** run `BC_DRY_RUN=1 jupyter nbconvert --to notebook --execute BC_prompt_probes_colab.ipynb --output-dir outputs/BC_DRYRUN_code_check_only`, or set `DRY_RUN = True` in the first cell. This mode uses Qwen3-0.6B and **random** vectors. It writes to `outputs/BC_DRYRUN_code_check_only/`, and every figure is titled "CODE CHECK ONLY: not results".

## Method (fixed by Flavia; do not change without asking)

- **Conventions copied from notebook 03:** Qwen/Qwen3-8B @ `b968826d9c46dd6066d109eabc6255188de91218`, NF4 with double quant and fp16 compute. The readout is the decoder-block output (before the next block). Block 23 is the main block. Stories are pooled from token index 49. Vectors are training means minus the equal-weight mean of means, then L2-normalized. The label and layer order are read from the files.
- **Part A score:** `(pooled[:, L] - center[L]) @ unit_vectors[:, L].T`. The notebook reports 14-way top-1 accuracy and one-vs-rest AUROC per emotion on the validation split (topics 14–16) at all 36 blocks.
- **Part A null:** shuffle the labels among the training stories, rebuild the vectors exactly as notebook 03 does, and rescore validation. 500 permutations, seed 180. The real value is compared with the null's 95th percentile, and p = (1 + #null ≥ real) / 501, so the smallest possible p is 0.002.
- **Part A test** (topics 17–19): used once, at block 23 and at the best validation block. The notebook reports accuracy, per-emotion AUROC, a confusion matrix, and a true-emotion × vector heatmap of mean test scores, z-scored per vector with validation mean and SD.
- **Part B/C:** the prompt is `apply_chat_template([user], add_generation_prompt=True, enable_thinking=False)`. The readout is the block-23 state at the last prompt token, which is the `\n\n` after the empty `<think></think>` block. The score is `cos(h − neutral_baseline, unit_vectors[e, 23])`, where the baseline is the mean over the 20 neutral prompts.

**Implementation choices made while coding (open to change):**
- The best validation block is the one with the highest validation top-1, with ties broken by macro-AUROC and then by the lower block.
- The null loop uses a vectorized Mann–Whitney AUROC in which tied scores share their average rank. It is tested against sklearn.
- `outputs/A/test_lock.json` records the blocks scored on test. A rerun that would test different blocks stops.
- Population SD is used for the heatmap z-scores.

## Tests run (2026-11-17)

| Test | Result | Where |
| --- | --- | --- |
| Archive keys and shapes: `pooled` (1120, 36, 4096) float32; vectors (14, 36, 4096); `center` (36, 4096) | pass | A §2 |
| Label order from file (alphabetical: afraid … surprised); layer order 0–35 is the same in both files; story order matches `stories.jsonl` (ids, emotions, splits) | pass | A §2 |
| Splits 784 / 168 / 168 with 56 / 12 / 12 per emotion; topics 0–13 / 14–16 / 17–19 | pass | A §2 |
| **Vectors rebuilt from the training rows match the saved arrays** (`emotion_means`, `center`, `raw_vectors`, `unit_vectors`) | allclose; max abs diff **0** for all four | A §3 |
| Synthetic point `center + 10·u_e` scores highest on *e* at every block, with score 10 | pass | A §3 |
| Scores have shape (1120, 36, 14); spot check against an explicit dot product | pass | A §3 |
| Fast AUROC = sklearn on a synthetic case with ties and on all 504 real validation columns | pass (atol 1e-12) | A §4, §6 |
| Training-split sanity (in-sample): top-1 well above chance | 0.463 at block 23 (6.5× chance); range 0.242–0.499 over blocks | A §5 |
| The null code path with unshuffled labels reproduces the real validation metrics | pass | A §7 |
| B/C forbidden-term checker catches inflections ("Calmly", "Loved", "sadness") and ignores look-alikes ("made", "average", "joystick") | pass | BC §3 |
| 115 texts (35 B, 20 neutral, 60 filled-in C prompts) contain none of the 101 forbidden terms | pass | BC §3 |
| Chat template ends `<|im_start|>assistant\n<think>\n\n</think>\n\n`; tokenization round-trips; last token `'\n\n'` | pass | BC §5 (dry run) |
| Hook returns shape `(hidden,)` (1024 for 0.6B) | pass | BC §5 (dry run) |
| Logits are unchanged with the hook attached and after removal; hook removed; hooked state = `hidden_states[24]` | max diffs 0.0 | BC §5 (dry run) |
| Dry run repeated under transformers 4.57.6 (the extraction run's version; Colab pins <5) and 5.20.0 | both pass; identical B/C outputs | BC (dry run) |
| Combined notebook, default laptop run (no CUDA): every cell runs; all 15 B/C cells print "skipped" | pass (34 s) | ABC |
| Combined notebook, code-check run (`BC_MODE=dryrun`): every cell runs; the Part A numbers equal those quoted in its "what it means" text (checked in code) | pass (56 s) | ABC |

Note: on transformers 5.x, `output_hidden_states=True` leaves a library output-capture hook on each block. The removal test therefore checks that *our* hook id is gone and that the block's hook set is back to what it was before. With 4.57.6 there are no library hooks.

## Results so far

### Part A: real results (Qwen3-8B, run `ab1333f49024705c`)

All numbers come from `outputs/A/summary.json`, `outputs/A/validation_by_block.csv` and `outputs/A/validation_auroc_vs_null_block23.csv`, written by `A_heldout_stories.ipynb` on 2026-11-17. Chance top-1 = 1/14 = 0.071.

**Validation (168 stories):**
- **Block 23:** top-1 is **0.292** against a null 95th percentile of 0.119 (null mean 0.073, p = 0.002). Macro AUROC is **0.796** against a null 95th percentile of 0.561.
- **Best validation block, 21:** top-1 is **0.321** against a null 95th percentile of 0.113. Macro AUROC is 0.792.
- **Every block:** top-1 is above the null 95th percentile at all **36 of 36** blocks. It rises from 0.19 at block 0 to a peak at block 21, then falls to 0.20 at block 35.
- **Desperate:** AUROC **0.842** at block 23 against a null 95th percentile of 0.691.
- **Calm:** AUROC **0.956** at block 23 against a null 95th percentile of 0.756.
- **Not above the null at block 23:** **frustrated** (0.654 vs 0.674) and **hopeful** (0.616 vs 0.695). Sad (0.681 vs 0.636) and guilty (0.776 vs 0.759) are only just above it.

**Test (168 stories, evaluated once at blocks 23 and 21):**
- **Block 23:** top-1 **0.232** (39/168) and macro AUROC **0.802**. Desperate AUROC is 0.786 and calm AUROC is 0.860.
- **Block 21:** top-1 0.232 and macro AUROC 0.797.

**What the test confusion matrix and heatmap show:**
- **Argmax predictions concentrate in a few columns.** At block 23, 32 of the 168 test stories are predicted *inspired* and 32 *surprised*.
- **Correct top-1 for desperate is rare.** Only 1 of its 12 test stories is predicted *desperate*, even though its AUROC is 0.786.
- **The z-scored heatmap still separates the emotions.** It shows a positive diagonal and a negative block (afraid, angry, desperate, frustrated, nervous) apart from a positive block (happy, hopeful, inspired, loving, calm).

**Caveats:**
- **In-sample training accuracy is only 0.46.** So even the training stories are often closer to another emotion's vector under this raw-projection argmax. Per-vector score offsets probably play a part (see the heatmap).
- **The null 95th percentiles for single-emotion AUROC are high (0.63–0.79).** Each emotion's 12 validation stories form a tight cluster, so even random training-mean directions separate them fairly often.
- **The labels are intended labels.** Every story is still tagged `pending_human_review`.

Figures (in `outputs/A/`, not committed until approved):
- `val_accuracy_by_block.png`
- `val_auroc_by_block_desperate_calm.png`
- `val_auroc_per_emotion_block23.png` and `val_auroc_per_emotion_block21.png`
- `null_accuracy_hist_block23.png`
- `test_confusion_heatmap_block23.png` and `test_confusion_heatmap_block21.png`

### Parts B and C: pending

These have not been run on Qwen3-8B yet; they need Colab. `prompts.json` is a draft awaiting review. The dry-run outputs (`outputs/BC_DRYRUN_code_check_only/`) come from Qwen3-0.6B with **random** vectors, so they are a code check only, not results. They do show one thing worth planning for: with random vectors, Spearman |ρ| reached 0.96 on some lines, **including the grapes control**. Monotone drift with X can happen without any emotion signal, so the C plots need the control (and possibly another reference) for interpretation.

## Outputs and git

`outputs/` is ignored. Small figures and CSV/JSON may be committed once approved (`git add -f validation_generalization/outputs/A/<file>`). Committed notebooks have their outputs cleared. The executed copies live in `outputs/` (`outputs/A/A_heldout_stories.executed.ipynb` and `outputs/BC_DRYRUN_code_check_only/*.executed.ipynb`).
