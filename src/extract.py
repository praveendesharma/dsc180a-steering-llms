"""Build emotion vectors for any Qwen3 model and test them on held-out stories.

Follows emotion_vector_pipeline/03_extract_emotion_vectors.ipynb: plain story text, no chat wrapper,
residual stream after each block, averaged from the 50th token on; each emotion's vector is its
mean over training stories minus the equal-weight average of all emotion means.
"""
import numpy as np
import torch

START_TOKEN_INDEX = 49  #the 50th token, as in notebook 03


def pooled_activations(model, tok, texts, start=START_TOKEN_INDEX):
    """Mean residual stream from token 50 on, after every block. Returns [n_texts, n_blocks, hidden] float32."""
    blocks = model.model.layers
    current, out = {}, []

    def make_hook(index):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            current[index] = hidden[0, start:].float().mean(dim=0).cpu()
        return hook

    handles = [block.register_forward_hook(make_hook(i)) for i, block in enumerate(blocks)]
    try:
        with torch.inference_mode():
            for t in texts:
                ids = tok(t, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
                assert ids.shape[1] > start, "story is shorter than 50 tokens"
                model.model(input_ids=ids, use_cache=False)  #the blocks only; no next-word head needed
                out.append(torch.stack([current[i] for i in range(len(blocks))]))
    finally:
        for h in handles:
            h.remove()
    return torch.stack(out).numpy().astype(np.float32)


def emotion_vectors(acts, emotions, splits, names):
    """Train-only emotion vectors. Returns (raw [E, L, d], unit [E, L, d], center [L, d])."""
    emotions, splits = np.asarray(emotions), np.asarray(splits)
    means = np.stack([acts[(emotions == e) & (splits == "train")].mean(axis=0) for e in names])
    center = means.mean(axis=0)
    raw = means - center[None]
    unit = raw / np.linalg.norm(raw, axis=-1, keepdims=True)
    return raw, unit, center


def projections(acts, unit, center):
    """How far each story points along each emotion vector, at every block: [n_stories, n_blocks, n_emotions]."""
    return np.einsum("nld,eld->nle", acts - center[None], unit)


def probe_accuracy(acts, labels, unit, center):
    """Label each story with the emotion whose vector it points along most; accuracy at every block."""
    pred = projections(acts, unit, center).argmax(axis=-1)
    return (pred == np.asarray(labels)[:, None]).mean(axis=0)
