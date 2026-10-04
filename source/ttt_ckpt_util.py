
from __future__ import annotations

import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "third_party" / "ttt_mlp_125m"


def repair_rope(model, device) -> int:
    
    n = 0
    for layer in model.model.layers:
        r = layer.seq_modeling_block.rotary_emb
        iv = 1.0 / (r.base ** (torch.arange(0, r.dim, 2, dtype=torch.int64)
                               .float() / r.dim))
        r.register_buffer("inv_freq", iv.to(device), persistent=False)
        n += 1
    got = model.model.layers[0].seq_modeling_block.rotary_emb.inv_freq[0].item()
    if abs(got - 1.0) > 1e-6:
        raise RuntimeError("inv_freq[0]=%r after repair, expected 1.0" % got)
    return n


def repair_token_idx(model, device) -> int:
    
    n = 0
    for layer in model.model.layers:
        sb = layer.seq_modeling_block
        t = 1.0 / torch.arange(1, sb.mini_batch_size + 1, dtype=torch.float32)
        sb.register_buffer("token_idx", t.to(device), persistent=False)
        n += 1
    got = model.model.layers[0].seq_modeling_block.token_idx
    if abs(float(got[0]) - 1.0) > 1e-6 or abs(float(got[1]) - 0.5) > 1e-6:
        raise RuntimeError("token_idx=%r after repair, expected 1/t"
                           % got[:3].tolist())
    return n


def _install_tied_weights_shim(embedding="model.embed_tokens.weight"):
    
    from transformers.modeling_utils import PreTrainedModel

    original = PreTrainedModel.get_expanded_tied_weights_keys
    if getattr(original, "_ttt_shimmed", False):
        return False

    def shimmed(self, *args, **kwargs):
        keys = getattr(type(self), "_tied_weights_keys", None)
        if isinstance(keys, (list, tuple)):
            type(self)._tied_weights_keys = {k: embedding for k in keys}
        return original(self, *args, **kwargs)

    shimmed._ttt_shimmed = True
    PreTrainedModel.get_expanded_tied_weights_keys = shimmed
    return True


def load_ttt(device="cuda", dtype=torch.float32, repair=True,
             model_dir=None):
    
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _install_tied_weights_shim()
    model_dir = MODEL_DIR if model_dir is None else model_dir
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True,
                                        local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True,
        dtype=dtype).to(device).eval()
    if repair:
        repair_rope(model, device)
        repair_token_idx(model, device)
    return model, tok


if __name__ == "__main__":
    m, _ = load_ttt(repair=False)
    bad = m.model.layers[0].seq_modeling_block.rotary_emb.inv_freq[:4].tolist()
    repair_rope(m, "cuda")
    good = m.model.layers[0].seq_modeling_block.rotary_emb.inv_freq[:4].tolist()
    print("as loaded :", bad)
    print("repaired  :", good)
