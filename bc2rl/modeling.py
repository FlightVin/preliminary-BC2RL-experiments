"""Model loading, LoRA adapters, sampling and batched token log-probabilities."""
from __future__ import annotations

import hashlib
import json
import os

import torch

from .losses import token_logprobs

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def model_info(proj_data):
    return json.load(open(os.path.join(proj_data, "model_revision.json")))


def load_tokenizer(path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_policy(path, lora_cfg: dict, seed: int, device="cuda"):
    """Base model in BF16 with a fresh LoRA adapter (B=0, so outputs equal the base model)."""
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    base = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa", local_files_only=True)
    base.to(device)
    for p in base.parameters():
        p.requires_grad_(False)
    torch.manual_seed(seed)
    cfg = LoraConfig(r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"], lora_dropout=lora_cfg["dropout"],
                     target_modules=lora_cfg.get("targets", LORA_TARGETS), bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(base, cfg)
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()          # adapter weights and optimizer state in FP32
    return model


def trainable_summary(model):
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"n_tensors": len(names), "n_params": int(count),
            "modules": sorted({n.split(".lora_")[0].split(".")[-1] for n in names})}


def adapter_state(model):
    from peft import get_peft_model_state_dict
    return {k: v.detach().to("cpu").contiguous() for k, v in get_peft_model_state_dict(model).items()}


def save_adapter(model, path):
    """Atomic write of adapter weights; returns sha256 of the file."""
    from safetensors.torch import save_file
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    save_file(adapter_state(model), tmp)
    os.replace(tmp, path)
    return file_sha256(path)


def load_adapter(model, path):
    from safetensors.torch import load_file
    from peft import set_peft_model_state_dict
    sd = load_file(path)
    res = set_peft_model_state_dict(model, sd)
    unexpected = getattr(res, "unexpected_keys", [])
    if unexpected:
        raise RuntimeError(f"unexpected adapter keys: {unexpected[:5]}")
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return file_sha256(path)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def chat_prompt(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)


def eos_ids(tok):
    ids = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    return sorted(i for i in ids if i is not None and i >= 0)


def decoding_config(tok, max_new_tokens):
    """Explicit sampling settings. Never inherit the model's generation_config defaults
    (Qwen ships temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.05)."""
    from transformers import GenerationConfig
    return GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=0, min_p=None,
                            repetition_penalty=1.0, max_new_tokens=max_new_tokens,
                            eos_token_id=eos_ids(tok), pad_token_id=tok.pad_token_id,
                            num_beams=1, use_cache=True)


DECODING_RECORD = {"temperature": 1.0, "top_p": 1.0, "top_k": "disabled", "repetition_penalty": 1.0,
                   "do_sample": True, "stop": "eos (<|im_end|>, <|endoftext|>)"}


@torch.no_grad()
def sample(model, tok, prompt_texts, n_per_prompt, max_new_tokens, seed, batch_seqs=512, adapter_off=False):
    """Sample n_per_prompt independent completions per prompt (prompt-major order).

    Returns list of dicts: prompt_index, attempt, prompt_ids, completion_ids, text, finish, n_tokens.
    """
    model.eval()
    tok.padding_side = "left"
    gen_cfg = decoding_config(tok, max_new_tokens)
    stops = set(eos_ids(tok))
    rows = [(i, a) for i in range(len(prompt_texts)) for a in range(n_per_prompt)]
    out = []
    for chunk_i, start in enumerate(range(0, len(rows), batch_seqs)):
        chunk = rows[start:start + batch_seqs]
        texts = [prompt_texts[i] for i, _ in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        torch.manual_seed(seed * 100003 + chunk_i)
        if adapter_off:
            with model.disable_adapter():
                gen = model.generate(**enc, generation_config=gen_cfg)
        else:
            gen = model.generate(**enc, generation_config=gen_cfg)
        plen = enc["input_ids"].shape[1]
        for r, (i, a) in enumerate(chunk):
            p_ids = enc["input_ids"][r][enc["attention_mask"][r].bool()].tolist()
            c = gen[r, plen:].tolist()
            finish, cut = "length", len(c)
            for t, tid in enumerate(c):
                if tid in stops:
                    finish, cut = "stop", t + 1
                    break
            c = c[:cut]
            text = tok.decode([t for t in c if t not in stops], skip_special_tokens=True)
            out.append({"prompt_index": i, "attempt": a, "prompt_ids": p_ids, "completion_ids": c,
                        "text": text, "finish": finish, "n_tokens": len(c)})
    return out


def build_batch(seqs, pad_id, device):
    """seqs: list of (prompt_ids, completion_ids). Right-padded input_ids, attention mask, and a
    completion mask aligned with the shifted targets (input_ids[:, 1:])."""
    L = max(len(p) + len(c) for p, c in seqs)
    B = len(seqs)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    att = torch.zeros((B, L), dtype=torch.long)
    cm = torch.zeros((B, L - 1), dtype=torch.float32)
    for b, (p, c) in enumerate(seqs):
        n = len(p) + len(c)
        ids[b, :n] = torch.tensor(p + c)
        att[b, :n] = 1
        cm[b, len(p) - 1:n - 1] = 1.0           # target positions that are completion tokens
    return ids.to(device), att.to(device), cm.to(device)


def seq_logprobs(model, seqs, pad_id, micro=8, grad=False, want_entropy=False):
    """Yield per-micro-batch (index slice, logp (b, L-1), mask, entropy or None)."""
    for s in range(0, len(seqs), micro):
        part = seqs[s:s + micro]
        ids, att, cm = build_batch(part, pad_id, model.device)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            logits = model(input_ids=ids, attention_mask=att).logits[:, :-1]
            lp, ent = token_logprobs(logits, ids[:, 1:], want_entropy)
        yield slice(s, s + len(part)), lp, cm, ent


def completion_logprob_table(model, seqs, pad_id, Lmax, micro=8):
    """No-grad pass: per-row completion log-probs left-aligned into (B, Lmax), plus mean entropy."""
    B = len(seqs)
    table = torch.zeros((B, Lmax), dtype=torch.float32, device=model.device)
    ent_sum, ent_n = 0.0, 0.0
    for sl, lp, cm, ent in seq_logprobs(model, seqs, pad_id, micro, grad=False, want_entropy=True):
        for r in range(lp.shape[0]):
            idx = cm[r].nonzero().squeeze(-1)
            table[sl.start + r, :len(idx)] = lp[r, idx]
        ent_sum += float((ent * cm).sum())
        ent_n += float(cm.sum())
    return table, ent_sum / max(ent_n, 1.0)


def gather_completion(lp, cm, Lmax):
    """Left-align completion log-probs of a micro-batch into (b, Lmax) keeping the graph."""
    rows = []
    for r in range(lp.shape[0]):
        idx = cm[r].nonzero().squeeze(-1)
        v = lp[r, idx]
        rows.append(torch.nn.functional.pad(v, (0, Lmax - v.shape[0])))
    return torch.stack(rows)
