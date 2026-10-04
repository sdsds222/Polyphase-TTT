from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from qa_data import load_hotpot, load_squad
from ttt_ckpt_util import load_ttt
from ttt_polyphase_real_binding import (
    build_groups,
    build_hotpot_groups,
    losses_for_ids,
    prediction,
)


ROOT = Path(__file__).resolve().parent
NAMES = ("base0", "base4", "base8", "base12")
RHO = 0.25
THRESHOLD = 0.04


def probabilities(losses):
    return [math.exp(-float(value)) for value in losses]


def margin(values):
    total = sum(values)
    ordered = sorted((value / total for value in values), reverse=True)
    return ordered[0] - ordered[1]


def average(items):
    return [sum(values) / len(values) for values in zip(*items)]


def fused(anchor, center):
    return [(1.0 - RHO) * left + RHO * right for left, right in zip(anchor, center)]


def run_mode(model, groups, arms, mode):
    predictions = []
    gold = []
    phase_counts = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for group in groups:
        for query_index, prompt in enumerate(group["prompts"]):
            ids = torch.tensor(
                [model._polyphase_tokenizer.encode(prompt, add_special_tokens=False)],
                device="cuda",
            )
            base_losses = losses_for_ids(model, ids, group["answer_ids"], arms["base0"])
            base_probs = probabilities(base_losses)
            if mode.startswith("baseline"):
                answer = prediction(base_losses)
                count = 1
            elif mode == "full":
                all_probs = [base_probs]
                for name in NAMES[1:]:
                    all_probs.append(probabilities(
                        losses_for_ids(model, ids, group["answer_ids"], arms[name])
                    ))
                answer = max(range(len(base_probs)), key=fused(base_probs, average(all_probs)).__getitem__)
                count = 4
            elif margin(base_probs) >= THRESHOLD:
                answer = prediction(base_losses)
                count = 1
            else:
                opposite_losses = losses_for_ids(model, ids, group["answer_ids"], arms["base8"])
                opposite_probs = probabilities(opposite_losses)
                if prediction(base_losses) == prediction(opposite_losses):
                    answer = max(
                        range(len(base_probs)),
                        key=fused(base_probs, average((base_probs, opposite_probs))).__getitem__,
                    )
                    count = 2
                else:
                    plus_probs = probabilities(
                        losses_for_ids(model, ids, group["answer_ids"], arms["base4"])
                    )
                    minus_probs = probabilities(
                        losses_for_ids(model, ids, group["answer_ids"], arms["base12"])
                    )
                    center = average((base_probs, plus_probs, opposite_probs, minus_probs))
                    answer = max(range(len(base_probs)), key=fused(base_probs, center).__getitem__)
                    count = 4
            predictions.append(answer)
            gold.append(query_index)
            phase_counts.append(count)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "predictions": predictions,
        "accuracy": sum(left == right for left, right in zip(predictions, gold)) / len(gold),
        "elapsed_seconds": elapsed,
        "questions_per_second": len(gold) / elapsed,
        "mean_phase_count": sum(phase_counts) / len(phase_counts),
        "phase_count_histogram": {str(value): phase_counts.count(value) for value in (1, 2, 4)},
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("squad", "hotpotqa"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--groups-per-kind", type=int, default=2)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model, tokenizer = load_ttt(dtype=torch.float32, model_dir=args.model_dir)
    model._polyphase_tokenizer = tokenizer
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    kinds = ("proper", "number", "other")
    if args.dataset == "hotpotqa":
        groups = build_hotpot_groups(
            tokenizer,
            load_hotpot(ROOT / "data/hotpotqa/hotpot_dev_distractor_v1.json"),
            3 * args.groups_per_kind,
            args.length,
            args.seed,
            answer_kinds=kinds,
            groups_per_kind=args.groups_per_kind,
        )
    else:
        contexts, questions = load_squad(ROOT / "data/squad/dev-v1.1.json")
        groups = build_groups(
            tokenizer,
            contexts,
            questions,
            3 * args.groups_per_kind,
            args.length,
            args.seed,
            answer_kinds=kinds,
            groups_per_kind=args.groups_per_kind,
        )
    heads = model.config.num_attention_heads
    arms = {
        "base0": None,
        "base4": {"offsets": [4] * heads},
        "base8": {"offsets": [8] * heads},
        "base12": {"offsets": [12] * heads},
    }
    first = torch.tensor(
        [[*tokenizer.encode(groups[0]["prompts"][0], add_special_tokens=False)]],
        device="cuda",
    )
    losses_for_ids(model, first, groups[0]["answer_ids"], None)
    results = {}
    for mode in ("baseline_pre", "gated", "full", "baseline_post"):
        results[mode] = run_mode(model, groups, arms, mode)
        print(mode, {key: value for key, value in results[mode].items() if key != "predictions"}, flush=True)
    baseline_seconds = (
        results["baseline_pre"]["elapsed_seconds"]
        + results["baseline_post"]["elapsed_seconds"]
    ) / 2
    results["gated"]["latency_multiple"] = results["gated"]["elapsed_seconds"] / baseline_seconds
    results["full"]["latency_multiple"] = results["full"]["elapsed_seconds"] / baseline_seconds
    results["gated"]["prediction_match_full"] = sum(
        left == right
        for left, right in zip(results["gated"]["predictions"], results["full"]["predictions"])
    ) / len(results["full"]["predictions"])
    payload = {
        "config": vars(args) | {"threshold": THRESHOLD, "residual_strength": RHO},
        "questions": 4 * len(groups),
        "results": results,
    }
    payload["config"]["model_dir"] = str(payload["config"]["model_dir"])
    payload["config"]["out"] = str(payload["config"]["out"])
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "gated_latency_multiple": results["gated"]["latency_multiple"],
        "full_latency_multiple": results["full"]["latency_multiple"],
        "gated_mean_phase_count": results["gated"]["mean_phase_count"],
        "gated_prediction_match_full": results["gated"]["prediction_match_full"],
    }, indent=2))
    print("saved", args.out)


if __name__ == "__main__":
    main()
