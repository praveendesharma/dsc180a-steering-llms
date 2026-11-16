# Qwen3-8B Emotion Vector Pipeline

Member 1’s (Praveen) dataset and vector-construction pipeline for the DSC 180A replication of [*Emotion Concepts and their Function in a Large Language Model*](https://arxiv.org/abs/2604.07729), using Qwen3-8B.

The pipeline generates stories that convey emotions indirectly, supports manual inspection of random samples, extracts residual-stream activations, and constructs one direction per emotion at every decoder block. These artifacts support the team’s validation, geometry, context, and steering experiments.

> **Start with the precomputed artifacts.** The full dataset, activations, and emotion vectors are available for downstream experiments. The notebooks document their construction and support reproduction; teammates do not need to rerun generation or extraction to use the saved results.
>
> **Story generation took approximately five hours in the original run.** Rerunning generation is only necessary when deliberately creating a new dataset. Inspection can review the existing stories independently; extraction is needed when recomputing activations or vectors. Runtime varies with hardware and retries.

## Repository structure

Only the following files are committed to GitHub:

```text
emotion_vector_pipeline/
├── README.md
├── 01_generate_emotion_stories_colab.ipynb
├── 02_inspect_emotion_stories.ipynb
├── 03_extract_emotion_vectors.ipynb
└── data/
    └── qwen3_8b_emotion_stories_1120_v1.zip
```

The large activation/vector ZIP is hosted separately on Google Drive. Extracted files and generated outputs are not additional repository uploads.

### Notebooks

| Notebook | Role | Environment |
| --- | --- | --- |
| `01_generate_emotion_stories_colab.ipynb` | Generates stories, validates responses, retries rejected batches, and saves resumable checkpoints. | Colab; NVIDIA CUDA GPU. |
| `02_inspect_emotion_stories.ipynb` | Draws reproducible random samples, collects manual judgments, and summarizes quality. | Jupyter or Colab; CPU. |
| `03_extract_emotion_vectors.ipynb` | Validates the full dataset, extracts pooled activations, constructs training-only vectors, and exports results. | Jupyter or Colab; NVIDIA CUDA GPU. |

### Precomputed results

[Download activations and vectors from Google Drive](https://drive.google.com/file/d/1L0kDEvq-fdpbWRxFUrWswGPEDSkiFMpu/view?usp=sharing)

Archive: `qwen3_8b_emotion_activations_vectors_ab1333f49024705c.zip` (approximately 1 GB). The archive corresponds to extraction run `ab1333f49024705c`. Its contents belong in `outputs/emotion_extraction/ab1333f49024705c/`. Model weights are not included. The link is a placeholder until the shared archive URL is available.

The extraction file listing reflects the notebook’s export format. The large results archive was not available for independent inspection.

## Optional local working structure

This layout shows where archive contents and notebook-generated outputs can live locally. It is **not the GitHub file list**. Teammates only need the files relevant to their experiment; the complete layout is unnecessary for loading existing vectors.

```text
emotion_vector_pipeline/
├── data/
│   ├── emotion_stories_qwen_full1120.jsonl   # Working dataset for notebook input
│   └── generation_v1/                      # Extracted dataset ZIP
│       ├── emotion_stories_qwen_full1120.jsonl
│       ├── manifest.json
│       ├── attempts.jsonl
│       └── summary.json
└── outputs/
    ├── story_review/                       # Created by manual inspection
    │   └── story_review_<dataset_hash>/
    │       ├── sample.json
    │       ├── ratings.json
    │       ├── review_details.csv
    │       ├── review_summary.csv
    │       └── story_manual_review.zip
    └── emotion_extraction/
        └── ab1333f49024705c/                # Extracted Drive ZIP or local run
            ├── story_activations.npz
            ├── emotion_vectors.npz
            ├── stories.jsonl
            ├── manifest.json
            ├── environment.json
            ├── train_counts.json
            ├── README.txt
            ├── cosine_similarity.png
            └── checkpoints/
                └── <story_id_hash>.npz
```

The dataset ZIP contains its four files at the archive root; `data/generation_v1/` is a local extraction destination. The JSONL directly under `data/` is the working copy used by notebook defaults. Likewise, the results archive contents belong directly inside the run directory, without an extra nested run folder. Review directories appear only when inspection is run. A different dataset or extraction configuration produces a different run ID.

## Dataset

**1,120 stories = 14 emotions × 20 topics × 4 stories per emotion/topic pair.** All accepted stories in the dataset were generated with Qwen3-8B. The same topics are used across all emotions.

Emotions: `desperate`, `nervous`, `surprised`, `calm`, `angry`, `loving`, `sad`, `afraid`, `inspired`, `happy`, `guilty`, `proud`, `frustrated`, `hopeful`.

| Split | Topic IDs | Stories per emotion | Total stories |
| --- | --- | ---: | ---: |
| Train | 0–13 | 56 | 784 |
| Validation | 14–16 | 12 | 168 |
| Test | 17–19 | 12 | 168 |

Splits are assigned by topic, so a topic does not appear in multiple splits. Only training stories construct the vectors. Validation/test activations are exported for downstream evaluation; this pipeline does not itself evaluate generalization.

Each JSONL record contains `id`, `emotion`, `topic_id`, `topic`, `story_index`, `split`, and `text`, plus generation provenance and review metadata. `story_index` ranges from 0 to 3 within each emotion/topic pair. `emotion` is the intended label, not an independently verified measurement.

### Generation archive

| Internal file | Contents |
| --- | --- |
| `emotion_stories_qwen_full1120.jsonl` | Accepted dataset consumed by inspection and extraction. |
| `manifest.json` | Generator model revision, sampling settings, topics, and environment metadata. |
| `attempts.jsonl` | Raw generation responses and automated-check results, including rejected attempts. |
| `summary.json` | Coverage and split counts; reports 1,120 stories with no missing pairs. |

The prompt requests separate paragraphs with no continuity and asks the model to convey emotions through behavior, physical reactions, dialogue, thoughts, and context. Generation checks enforce formatting, listed prohibited terms, exact-duplicate rejection, and non-truncation. They do not establish emotional correctness or exhaustively detect synonyms and near duplicates.

**Provenance note:** the final generation code removes the word-count restriction. The archived manifest still contains `word_range: [100, 160]` and its original prompt-version identifier. Those fields are stale and should not be interpreted as enforced constraints. Observed accepted stories range from 69 to 159 whitespace-separated words.

## Workflow and requirements

The workflow is **generation → manual inspection → activation extraction → downstream experiments**. Precomputed data and vectors allow downstream experiments to begin without rerunning generation or extraction.

Generation and extraction use PyTorch, Transformers 4.x, Accelerate, and bitsandbytes with NF4 quantization and float16 computation. The recorded run used a Tesla T4. Inspection uses pandas and IPython and does not load a language model. GPU notebook compatibility is limited to the current NVIDIA CUDA configuration; CPU and Apple MPS extraction are not implemented.

In the inspection and extraction notebooks, `DATA_PATH` defaults to `data/emotion_stories_qwen_full1120.jsonl`. `OUTPUT_ROOT` controls the corresponding local output directory. Relative paths are resolved from the kernel’s working directory.

### Manual inspection outputs

The inspection notebook samples up to ten stories per emotion: 140 stories for this dataset. Reviewers first record their impression without seeing the intended label, then judge emotion fit, explicit emotion terms, premise fidelity, and optional notes. Completed judgments are saved after each story and resumed on rerun; typing `q` pauses the session.

| File | Contents |
| --- | --- |
| `sample.json` | Selected stories, sampling seed, and dataset fingerprint. |
| `ratings.json` | Completed per-story judgments and notes. |
| `review_details.csv` | Tabular judgments, keyed by story ID. |
| `review_summary.csv` | Per-emotion review counts and quality rates. |
| `story_manual_review.zip` | Export of the sample, judgments, and summaries for restoration or sharing. |

The paper manually inspected ten stories for thirty emotions. This project adapts that procedure to fourteen emotions; the blind first impression and scoring rubric are project additions. The sample is a dataset quality audit, not a held-out model evaluation. Test stories inspected during development should not subsequently be described as untouched test data.

## Extraction and vector construction

The model reads plain story text without an emotion label, topic, or chat wrapper. Hooks capture each decoder block’s output after residual additions and before the next block; the last block is captured before final RMSNorm.

For each story and block, the notebook averages activations from the **50th token inclusive** (Python index 49). Earlier tokens still provide context. Token lengths must exceed index 49 and may not exceed the configured 512-token limit; longer stories are rejected rather than silently truncated. Saved output from the recorded run reports 91–195 tokens per story.

At each block, the direction for an emotion is its mean training-story representation minus the equal-weight mean of all fourteen emotion means. Raw directions and unit-normalized directions are both saved. The selected reference block is index 23, the 24th of 36 blocks; all blocks remain available.

| Array | Axes | Supplied run shape |
| --- | --- | --- |
| Pooled story activations | story, block, hidden dimension | `(1120, 36, 4096)` |
| Emotion directions | emotion, block, hidden dimension | `(14, 36, 4096)` |

These are pooled representations, not complete token-by-token activations. Context/token-position experiments require additional extraction on their own prompts.

## Extraction archive and loading vectors

The notebook exports `story_activations.npz`, `emotion_vectors.npz`, `stories.jsonl`, `manifest.json`, `environment.json`, `train_counts.json`, `README.txt`, the similarity figure, and per-story `checkpoints/`.

Generation and extraction each have a `manifest.json`; the containing directory identifies the stage.

```python
from pathlib import Path
import numpy as np

run_dir = Path("outputs/emotion_extraction/ab1333f49024705c")
with np.load(run_dir / "emotion_vectors.npz", allow_pickle=False) as saved:
    emotions = saved["emotion_names"].tolist()
    layers = saved["layer_indices"].tolist()
    raw_vectors = saved["raw_vectors"].copy()
    unit_vectors = saved["unit_vectors"].copy()

happy_at_block_23 = raw_vectors[emotions.index("happy"), layers.index(23)]
```

Always read saved label and layer ordering. For downstream model experiments, match the extraction manifest’s model revision, tokenizer, quantization, and residual location. Generation and the recorded extraction output report model commit `b968826d9c46dd6066d109eabc6255188de91218`. Record the resolved commit rather than relying on a changing `main` revision. Seeds aid reproducibility but do not guarantee identical outputs across hardware/software versions.

Unit directions are convenient for geometry. Steering magnitudes must be calibrated for Qwen; coefficients from the original paper cannot be assumed to transfer.

## Team handoff and current scope

| Workstream | Starting artifacts |
| --- | --- |
| Validation/generalization | Vectors, held-out story activations, dataset split labels, extraction manifest. |
| Geometry/layer comparisons | Raw/unit vectors and emotion means across all blocks. |
| Context/token-position experiments | Vectors and hook convention; extract new token-level activations. |
| Causal steering | Vectors and a matching model configuration; implement and test interventions. |

The recorded extraction notebook outputs report successful full extraction and vector construction for run `ab1333f49024705c`. Every dataset record remains tagged `pending_human_review`; completed manual-review results are not included in this handoff. Automated extraction does not establish human verification.

This implements an adapted vector-construction pipeline, not the entire paper: Qwen3-8B replaces Claude, the dataset is smaller, weights use NF4 quantization, and neutral-transcript PCA correction is **not implemented**. No held-out performance or causal steering result is established by the cosine heatmap.

