"""Steer Qwen3-8B by adding an emotion vector to the residual stream while it generates.

Loads the model exactly like emotion_vector_pipeline/03_extract_emotion_vectors.ipynb
(same revision, NF4 quantization and hook location) so the saved vectors line up with it.
"""
from contextlib import contextmanager

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen3-8B"
#revision the vectors were extracted from (extraction run ab1333f49024705c)
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
#0-based block the team picked for the vectors (24th of 36)
LAYER = 23

#the small model from the playground; same "two-thirds of the way in" rule: round(2 * 28 / 3) - 1
SMALL_MODEL_ID = "Qwen/Qwen3-0.6B"
SMALL_LAYER = 18


def load_model_small(model_id=SMALL_MODEL_ID, revision=None):
    """Load Qwen3-0.6B in bfloat16 on the first CUDA GPU (small enough to skip quantization). Returns (tok, model)."""
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa",
    )
    model.eval()
    return tok, model


def load_model_8b(revision=MODEL_REVISION):
    """Load Qwen3-8B in 4-bit NF4 on the first CUDA GPU. Returns (tok, model)."""
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)

    #same quantization as notebook 03: NF4 weights, double quantization, float16 math
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=revision,
        quantization_config=quant,
        torch_dtype=torch.float16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.eval()
    return tok, model


def load_vector(path, emotion, layer=LAYER):
    """Read one unit-length emotion vector (shape [hidden]) from emotion_vectors.npz."""
    with np.load(path, allow_pickle=False) as saved:
        #read the saved orderings instead of guessing them
        names = saved["emotion_names"].tolist()
        layers = saved["layer_indices"].tolist()
        v = saved["unit_vectors"][names.index(emotion), layers.index(layer)].copy()
    return torch.tensor(v, dtype=torch.float32)


def _hidden(output):
    #decoder blocks return a tuple in some transformers versions and a tensor in others
    return output[0] if isinstance(output, tuple) else output


def residual_norms(model, tok, texts, layer=LAYER):
    """Per-token L2 norms of the residual stream at `layer`, split into (first token, the rest).

    The first token is kept apart because its norm is often far larger than every other
    token's (an "attention sink"), which would skew an average.
    """
    first, rest = [], []

    def hook(module, args, output):
        n = _hidden(output)[0].float().norm(dim=-1).cpu()
        first.append(n[:1])
        rest.append(n[1:])

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            for t in texts:
                ids = tok(t, return_tensors="pt").input_ids.to(model.device)
                model(ids)
    finally:
        handle.remove()
    return torch.cat(first), torch.cat(rest)


@contextmanager
def steer(model, vector, strength, norm, layer=LAYER):
    """While active, add strength × norm × vector to every token's residual stream at `layer`.

    strength works like the paper's: a fraction of the average residual-stream norm
    at that layer (the paper used 0.5). vector should be unit length.
    """
    add = (strength * norm * vector).to(model.device)

    def hook(module, args, output):
        hidden = _hidden(output)
        steered = hidden + add.to(hidden.dtype)
        return (steered,) + tuple(output[1:]) if isinstance(output, tuple) else steered

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def generate(model, tok, prompt, seed=0, max_new_tokens=80, temperature=0.7, top_p=0.9):
    """Sample a chat reply to `prompt` with a fixed seed. Same seed + same prompt = matched runs."""
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    inp = tok(text, return_tensors="pt").to(model.device)

    #reset the random generator so every strength samples with the same randomness
    torch.manual_seed(seed)
    with torch.inference_mode():
        out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=temperature, top_p=top_p, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
