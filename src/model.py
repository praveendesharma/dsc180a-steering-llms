"""Load Qwen3 and generate text from it.

Moved from notebooks/playground.ipynb so every notebook can reuse it.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3-0.6B"


def get_device():
    #gpu acceleration: cuda (nvidia), mps (apple silicon), or cpu
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(name=MODEL_NAME, device=None):
    """Download (first run only) and load the tokenizer and model. Returns (tok, model, device)."""
    device = device or get_device()

    #qwen's tokenizer
    tok = AutoTokenizer.from_pretrained(name)

    #qwen's model
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).to(device)

    #configure model for inference
    model.eval()
    return tok, model, device


def chat(model, tok, user_msg, max_new_tokens=60, sample=False):
    #initiate conversation.
    msgs = [{"role": "user", "content": user_msg}]

    #wrap message in markers qwen uses to identify user vs assistant messages.
    #add_generation_prompt=True hints the model to generate a response at the end of the input prompt.
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)

    #tokenize the input, get IDs Tensor. Same as before.
    inp = tok(prompt, return_tensors="pt").to(model.device)

    #start the autoregressive loop to predict token, append to its stream and repeat.
    #max_new_tokens=60 caps the number of output tokens @ 60.
    #do_sample=False makes the model deterministic, and it will only output the most likely token.
    #temperature=0.8 makes the model more creative, plays around with the perplexity of the next token
    gen = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=sample,
                         temperature=0.8 if sample else None)

    #extract 1st back-and-forth of the conversation, drop the input prompt tokens,
    # and decodes the output tokens from IDs to text
    return tok.decode(gen[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
