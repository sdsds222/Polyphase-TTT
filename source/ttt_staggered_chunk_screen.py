from __future__ import annotations

import argparse
import json
import math
import random
import types
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F

from ttt_ckpt_util import load_ttt


ROOT = Path(__file__).resolve().parent


def load_tokens(tokenizer, name: str) -> torch.Tensor:
    if name == "squad":
        raw = json.loads((ROOT / "data/squad/dev-v1.1.json").read_text(encoding="utf-8"))
        contexts = [p["context"] for a in raw["data"] for p in a["paragraphs"]]
        text = "\n\n".join(contexts[len(contexts) // 2:])
    else:
        text = (ROOT / "data" / name / "test.txt").read_text(encoding="utf-8")
    ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    return torch.tensor(ids, dtype=torch.long)


def segments(tokens: torch.Tensor, length: int, count: int, seed: int):
    rng = random.Random(seed)
    starts = sorted(rng.sample(range(tokens.numel() - length - 1), count))
    return starts, [tokens[s:s + length + 1] for s in starts]


def phase_token_eta(module, phases: torch.Tensor, ready: dict[str, torch.Tensor]):
    schedule = torch.clamp_min(module.token_idx + module.learnable_token_idx, 0.0)
    values = schedule[phases]
    b, h, n = ready["XV"].shape[:3]
    k = phases.numel()
    token_eta = values.reshape(1, 1, 1, k, 1).expand(b, h, n, k, 1)
    ready["token_eta"] = token_eta
    ready["eta"] = token_eta * ready["ttt_lr_eta"]
    return ready


def calendar_output(module, hidden_states, position_ids, offset, globals_):
    b, length = hidden_states.shape[:2]
    width = module.mini_batch_size
    xq, xk, xv = module.get_qkv_projections(hidden_states, cache_params=None)
    xq = xq.reshape(b, length, module.num_heads, module.head_dim).transpose(1, 2)
    xk = xk.reshape(b, length, module.num_heads, module.head_dim).transpose(1, 2)
    xv = xv.reshape(b, length, module.num_heads, module.head_dim).transpose(1, 2)
    phase = (position_ids - offset) % width
    cos, sin = module.rotary_emb(xv, phase)
    xq, xk = globals_["permute_qk"](xq, xk)
    xq, xk = globals_["apply_rotary_pos_emb"](xq, xk, cos, sin)
    xq, xk = globals_["undo_permute_qk"](xq, xk)

    cuts = [0]
    if offset:
        cuts.append(min(offset, length))
    cursor = cuts[-1]
    while cursor + width <= length:
        cursor += width
        cuts.append(cursor)
    if cuts[-1] != length:
        cuts.append(length)

    outputs = []
    state = None
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        size = hi - lo
        values = {
            "XQ": xq[:, :, lo:hi],
            "XK": xk[:, :, lo:hi],
            "XV": xv[:, :, lo:hi],
            "X": hidden_states[:, lo:hi],
        }
        ready = module.get_ttt_inputs(values, size, None)
        phases = ((torch.arange(lo, hi, device=hidden_states.device) - offset) % width).long()
        ready = phase_token_eta(module, phases, ready)
        output, state = module.ttt(
            ready,
            mini_batch_size=size,
            last_mini_batch_params_dict=state,
            cache_params=None,
        )
        outputs.append(output)
    return torch.cat(outputs, dim=1).reshape(b, length, module.num_heads, module.head_dim)


@contextmanager
def staggered_calendar(model, offsets, rotate_layers=False, layer_stride=0,
                       reverse_layers=False, average_offsets=None,
                       average_strength=1.0):
    saved = []
    for layer_index, layer in enumerate(model.model.layers):
        module = layer.seq_modeling_block
        original = module.forward
        globals_ = original.__func__.__globals__
        saved.append((module, original))

        def forward(self, hidden_states, attention_mask=None, position_ids=None,
                    cache_params=None, _g=globals_, _li=layer_index):
            if cache_params is not None:
                raise RuntimeError("staggered screen supports prefill without cache only")
            if average_offsets is not None:
                candidates = [
                    calendar_output(self, hidden_states, position_ids, value, _g)
                    for value in average_offsets
                ]
                mean = torch.stack(candidates, dim=0).mean(dim=0)
                combined = candidates[0] + average_strength * (mean - candidates[0])
                output = self.post_norm(combined.reshape(hidden_states.shape))
                if self.use_gate:
                    output = self.apply_gate(hidden_states, output)
                return self.o_proj(output)
            head_offsets = list(offsets)
            if layer_stride:
                layer_phase = (_li * layer_stride) % self.mini_batch_size
                if reverse_layers:
                    layer_phase = (-layer_phase) % self.mini_batch_size
                head_offsets = [layer_phase] * self.num_heads
            elif rotate_layers:
                shift = _li % len(head_offsets)
                head_offsets = head_offsets[shift:] + head_offsets[:shift]
            unique = sorted(set(head_offsets))
            candidates = {
                value: calendar_output(self, hidden_states, position_ids, value, _g)
                for value in unique
            }
            combined = torch.empty_like(next(iter(candidates.values())))
            for head, value in enumerate(head_offsets):
                combined[:, :, head] = candidates[value][:, :, head]
            output = self.post_norm(combined.reshape(hidden_states.shape))
            if self.use_gate:
                output = self.apply_gate(hidden_states, output)
            return self.o_proj(output)

        module.forward = types.MethodType(forward, module)
    try:
        yield
    finally:
        for module, original in saved:
            module.forward = original


def token_losses(logits, targets):
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
    ).reshape(targets.shape)


def summarize(losses, baseline, width=16, warmup=16):
    flat = losses[:, warmup:].flatten()
    base = baseline[:, warmup:].flatten()
    phases = torch.arange(flat.numel(), device=flat.device) % width
    by_phase = [float(flat[phases == p].mean()) for p in range(width)]
    boundary = (phases == 0) | (phases == width - 1)
    return {
        "nll": float(flat.mean()),
        "delta_pct": float((flat.mean() / base.mean() - 1) * 100),
        "boundary_delta_pct": float((flat[boundary].mean() / base[boundary].mean() - 1) * 100),
        "interior_delta_pct": float((flat[~boundary].mean() / base[~boundary].mean() - 1) * 100),
        "phase_mean_std": float(torch.tensor(by_phase).std(unbiased=False)),
        "wins": int((flat < base).sum()),
        "losses": int((flat > base).sum()),
        "by_phase": by_phase,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=ROOT / "third_party/ttt_mlp_125m")
    ap.add_argument("--datasets", default="squad,penn-treebank")
    ap.add_argument("--length", type=int, default=128)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--arms", default="all")
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--out", type=Path, default=ROOT / "ttt_staggered_chunk_screen.json")
    args = ap.parse_args()
    if args.length % 16:
        raise ValueError("length must be divisible by 16")

    model, tokenizer = load_ttt(dtype=torch.float32, model_dir=args.model_dir)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    heads = model.config.num_attention_heads
    base_stagger = ([0, 4, 8, 12] * math.ceil(heads / 4))[:heads]
    arms = {
        "noop": {"offsets": [0] * heads},
        "global4": {"offsets": [4] * heads},
        "stagger4": {"offsets": base_stagger},
        "stagger4_roll1": {"offsets": base_stagger[1:] + base_stagger[:1]},
        "stagger4_reverse": {"offsets": list(reversed(base_stagger))},
        "stagger2": {"offsets": ([0, 8] * math.ceil(heads / 2))[:heads]},
        "stagger4_rotate": {"offsets": base_stagger, "rotate_layers": True},
        "layer_cycle4": {"offsets": [0] * heads, "layer_stride": 4},
        "layer_cycle4_reverse": {
            "offsets": [0] * heads, "layer_stride": 4, "reverse_layers": True
        },
        "layer_cycle2": {"offsets": [0] * heads, "layer_stride": 8},
        "poly2": {"offsets": [0] * heads, "average_offsets": [0, 8]},
        "poly4": {"offsets": [0] * heads, "average_offsets": [0, 4, 8, 12]},
        "poly4half": {
            "offsets": [0] * heads,
            "average_offsets": [0, 4, 8, 12],
            "average_strength": 0.5,
        },
    }
    if args.arms != "all":
        selected = {value.strip() for value in args.arms.split(",")}
        arms = {name: settings for name, settings in arms.items() if name in selected}
        missing = selected - set(arms)
        if missing:
            raise ValueError(f"unknown arms: {sorted(missing)}")
    result = {"config": vars(args) | {"model_dir": str(args.model_dir), "out": str(args.out)}, "datasets": {}}
    checked = False
    with torch.inference_mode():
        for dataset_index, name in enumerate(x.strip() for x in args.datasets.split(",")):
            starts, windows = segments(load_tokens(tokenizer, name), args.length, args.samples,
                                       args.seed + 1009 * dataset_index)
            rows = {arm: [] for arm in arms}
            for sample, window in enumerate(windows):
                x = window[:-1][None].cuda()
                y = window[1:][None].cuda()
                base_logits = model(input_ids=x, use_cache=False, return_dict=True).logits
                base_loss = token_losses(base_logits, y)
                for arm, settings in arms.items():
                    with staggered_calendar(model, **settings):
                        logits = model(input_ids=x, use_cache=False, return_dict=True).logits
                    if arm == "noop" and not checked:
                        difference = (logits.float() - base_logits.float()).abs()
                        result["correctness"] = {
                            "max_abs_logit_difference": float(difference.max()),
                            "mean_abs_logit_difference": float(difference.mean()),
                        }
                        if float(difference.max()) > 3e-5:
                            raise RuntimeError("all-zero calendar is not an exact no-op")
                        checked = True
                    row = summarize(token_losses(logits, y), base_loss, warmup=args.warmup)
                    row["sample"] = sample
                    rows[arm].append(row)
                    print(name, sample, arm, {k: round(row[k], 5) for k in
                          ("delta_pct", "boundary_delta_pct", "interior_delta_pct", "phase_mean_std")}, flush=True)
            result["datasets"][name] = {"starts": starts, "arms": rows}
            args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("saved", args.out)


if __name__ == "__main__":
    main()
