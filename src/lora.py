"""Train a rank-1 LoRA adapter on one block's down_proj and read out what it learned.

The adapter adds B·(A·x) to down_proj's output, where x is that block's 12,288 MLP neuron
values for one token. A (1 × 12,288) is the "filter": A·x is how hard it fires on a token.
B (4,096 × 1) is the "steering" direction it writes into the residual stream, the same
space the emotion vectors live in, so B can be compared with them directly.
"""
import torch
from peft import LoraConfig, get_peft_model

from .steering import LAYER


def add_adapter(model, layer=LAYER, rank=1, alpha=1):
    """Freeze the model and attach a rank-`rank` LoRA to down_proj at `layer` only."""
    cfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
        target_modules=["down_proj"], layers_to_transform=[layer],
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, cfg)


def down_proj(model, layer=LAYER):
    #the LoRA-wrapped down_proj inside the peft model
    return model.base_model.model.model.layers[layer].mlp.down_proj


def adapter_weights(model, layer=LAYER):
    """Return (A, B, scaling): A has shape [12288], B has shape [4096] for rank 1."""
    mod = down_proj(model, layer)
    A = mod.lora_A["default"].weight.detach().float().cpu()[0]
    B = mod.lora_B["default"].weight.detach().float().cpu()[:, 0]
    return A, B, mod.scaling["default"]


def text_loss(model, tok, texts):
    """Average next-token loss over `texts` (plain text, no chat wrapper)."""
    total = 0.0
    with torch.inference_mode():
        for t in texts:
            ids = tok(t, return_tensors="pt").input_ids.to(model.device)
            total += model(input_ids=ids, labels=ids).loss.item()
    return total / len(texts)


def train_adapter(model, tok, texts, epochs=4, lr=1e-3, seed=0, log=print):
    """Plain next-token training on `texts`, one story per step. Only A and B get updates."""
    torch.manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    model.train()
    losses = []
    for epoch in range(epochs):
        for i in torch.randperm(len(texts)).tolist():
            ids = tok(texts[i], return_tensors="pt").input_ids.to(model.device)
            loss = model(input_ids=ids, labels=ids, use_cache=False).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append(loss.item())
        log(f"epoch {epoch + 1}: mean loss {sum(losses[-len(texts):]) / len(texts):.3f}")
    model.eval()
    return losses


def firing(model, tok, texts, layer=LAYER):
    """A·x for every token of every text (from token 50 on, like the vectors), one tensor per text."""
    mod = down_proj(model, layer)
    A = mod.lora_A["default"].weight
    out, current = [], {}

    def pre(module, args):
        x = args[0]
        current["z"] = (x.to(A.dtype) @ A.T)[0, :, 0].float().cpu()

    handle = mod.register_forward_pre_hook(pre)
    try:
        with torch.inference_mode():
            for t in texts:
                ids = tok(t, return_tensors="pt").input_ids.to(model.device)
                model(input_ids=ids)
                out.append(current["z"][49:])
    finally:
        handle.remove()
    return out
