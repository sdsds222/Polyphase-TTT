from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from ttt_ckpt_util import load_ttt
from ttt_staggered_chunk_screen import staggered_calendar


ROOT = Path(__file__).resolve().parent


def load_examples(path: Path, tokenizer, count: int, seed: int, max_tokens: int,
                  excluded=None):
    excluded = set() if excluded is None else set(excluded)
    source = [json.loads(line)["text"] for line in path.read_text(encoding="utf-8").splitlines()]
    indices = list(range(len(source)))
    random.Random(seed).shuffle(indices)
    rows = []
    for index in indices:
        if index in excluded:
            continue
        text = source[index].strip()
        if " " not in text:
            continue
        context, last = text.rsplit(" ", 1)
        target = " " + last
        context_ids = tokenizer.encode(context, add_special_tokens=False)
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        full_ids = tokenizer.encode(context + target, add_special_tokens=False)
        if not context_ids or not target_ids:
            continue
        if context_ids + target_ids != full_ids:
            continue
        if len(full_ids) > max_tokens:
            continue
        rows.append({
            "source_index": index,
            "context_ids": context_ids,
            "target_ids": target_ids,
            "target_text": target,
        })
        if len(rows) == count:
            break
    if len(rows) != count:
        raise RuntimeError(f"selected only {len(rows)} of {count} examples")
    return rows, len(source)


@torch.inference_mode()
def target_distribution(model, context_ids, target_ids, settings):
    combined = context_ids + target_ids
    ids = torch.tensor([combined], dtype=torch.long, device="cuda")
    if settings is None:
        logits = model(input_ids=ids, use_cache=False, return_dict=True).logits[0].float()
    else:
        with staggered_calendar(model, **settings):
            logits = model(input_ids=ids, use_cache=False, return_dict=True).logits[0].float()
    start = len(context_ids) - 1
    selected = logits[start:start + len(target_ids)]
    log_probabilities = selected.log_softmax(dim=-1)
    probabilities = log_probabilities.exp()
    targets = torch.tensor(target_ids, dtype=torch.long, device="cuda")
    gold_log_probabilities = log_probabilities.gather(-1, targets[:, None]).squeeze(-1)
    prediction = probabilities.argmax(dim=-1)
    return probabilities, gold_log_probabilities, prediction


def mean(values):
    return sum(values) / len(values)


def exact(prediction, target):
    return list(prediction) == list(target)


def paired_sign_p(removed: int, introduced: int):
    discordant = removed + introduced
    if discordant == 0:
        return 1.0
    lower = min(removed, introduced)
    tail = sum(math.comb(discordant, value) for value in range(lower + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def summarize(rows, phase_names, residual_strength):
    phase_accuracy = {
        name: mean([exact(row["phase_predictions"][name], row["target_ids"]) for row in rows])
        for name in phase_names
    }
    result = {
        "examples": len(rows),
        "phase_accuracy": phase_accuracy,
        "phase_mean_accuracy": mean(list(phase_accuracy.values())),
        "phase_worst_accuracy": min(phase_accuracy.values()),
        "phase_best_accuracy": max(phase_accuracy.values()),
        "phase_mean_gold_nll": {
            name: mean([row["phase_gold_nll"][name] for row in rows])
            for name in phase_names
        },
    }
    if len(phase_names) == 1:
        return result
    residual_accuracy = {
        name: mean([exact(row["residual_predictions"][name], row["target_ids"]) for row in rows])
        for name in phase_names
    }
    raw_prediction_flip = [
        len({tuple(row["phase_predictions"][name]) for name in phase_names}) > 1
        for row in rows
    ]
    residual_prediction_flip = [
        len({tuple(row["residual_predictions"][name]) for name in phase_names}) > 1
        for row in rows
    ]
    raw_correctness_flip = [
        len({exact(row["phase_predictions"][name], row["target_ids"]) for name in phase_names}) > 1
        for row in rows
    ]
    residual_correctness_flip = [
        len({exact(row["residual_predictions"][name], row["target_ids"]) for name in phase_names}) > 1
        for row in rows
    ]
    default_raw = [exact(row["phase_predictions"][phase_names[0]], row["target_ids"]) for row in rows]
    default_residual = [
        exact(row["residual_predictions"][phase_names[0]], row["target_ids"]) for row in rows
    ]
    prediction_removed = sum(a and not b for a, b in zip(raw_prediction_flip, residual_prediction_flip))
    prediction_introduced = sum(not a and b for a, b in zip(raw_prediction_flip, residual_prediction_flip))
    correctness_removed = sum(a and not b for a, b in zip(raw_correctness_flip, residual_correctness_flip))
    correctness_introduced = sum(not a and b for a, b in zip(raw_correctness_flip, residual_correctness_flip))
    default_corrected = sum(not a and b for a, b in zip(default_raw, default_residual))
    default_broken = sum(a and not b for a, b in zip(default_raw, default_residual))
    result.update({
        "residual_strength": residual_strength,
        "residual_phase_accuracy": residual_accuracy,
        "residual_phase_mean_accuracy": mean(list(residual_accuracy.values())),
        "residual_phase_worst_accuracy": min(residual_accuracy.values()),
        "raw_prediction_flip_rate": mean(raw_prediction_flip),
        "residual_prediction_flip_rate": mean(residual_prediction_flip),
        "prediction_flips_removed": prediction_removed,
        "prediction_flips_introduced": prediction_introduced,
        "prediction_flip_paired_p": paired_sign_p(prediction_removed, prediction_introduced),
        "raw_correctness_flip_rate": mean(raw_correctness_flip),
        "residual_correctness_flip_rate": mean(residual_correctness_flip),
        "correctness_flips_removed": correctness_removed,
        "correctness_flips_introduced": correctness_introduced,
        "correctness_flip_paired_p": paired_sign_p(correctness_removed, correctness_introduced),
        "default_raw_accuracy": mean(default_raw),
        "default_residual_accuracy": mean(default_residual),
        "default_corrected": default_corrected,
        "default_broken": default_broken,
        "default_accuracy_paired_p": paired_sign_p(default_corrected, default_broken),
    })
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=ROOT / "third_party/ttt_mlp_350m")
    ap.add_argument("--data", type=Path, default=ROOT / "data/lambada/lambada_test.jsonl")
    ap.add_argument("--mode", choices=("base", "phase"), default="base")
    ap.add_argument("--questions", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--exclude-seed", type=int, default=-1)
    ap.add_argument("--exclude-questions", type=int, default=0)
    ap.add_argument("--exclude-runs", default="")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--phase-offsets", default="0,4,8,12")
    ap.add_argument("--residual-strength", type=float, default=0.25)
    ap.add_argument("--out", type=Path, default=ROOT / "ttt_polyphase_lambada.json")
    args = ap.parse_args()

    model, tokenizer = load_ttt(dtype=torch.float32, model_dir=args.model_dir)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    excluded = set()
    if args.exclude_runs:
        for specification in args.exclude_runs.split(","):
            seed_text, count_text = specification.strip().split(":", 1)
            prior, _ = load_examples(
                args.data, tokenizer, int(count_text), int(seed_text),
                args.max_tokens, excluded=excluded
            )
            excluded.update(row["source_index"] for row in prior)
    if args.exclude_seed >= 0 and args.exclude_questions > 0:
        prior, _ = load_examples(
            args.data, tokenizer, args.exclude_questions, args.exclude_seed,
            args.max_tokens, excluded=excluded
        )
        excluded = {row["source_index"] for row in prior}
    selected, dataset_size = load_examples(
        args.data, tokenizer, args.questions, args.seed, args.max_tokens,
        excluded=excluded
    )
    heads = model.config.num_attention_heads
    if args.mode == "base":
        arms = {"base0": None}
    else:
        phase_offsets = [
            int(value.strip())
            for value in args.phase_offsets.split(",")
            if value.strip()
        ]
        if not phase_offsets or phase_offsets[0] != 0:
            raise ValueError("phase offsets must start with 0 so base0 remains the anchor")
        if len(set(phase_offsets)) != len(phase_offsets):
            raise ValueError("phase offsets must be unique")
        if any(offset < 0 or offset >= 16 for offset in phase_offsets):
            raise ValueError("phase offsets must be between 0 and 15")
        arms = {
            f"base{offset}": (
                None if offset == 0 else {"offsets": [offset] * heads}
            )
            for offset in phase_offsets
        }

    rows = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for index, sample in enumerate(selected):
        phase_probabilities = {}
        phase_predictions = {}
        phase_gold_nll = {}
        for name, settings in arms.items():
            probabilities, gold_log_probabilities, prediction = target_distribution(
                model, sample["context_ids"], sample["target_ids"], settings
            )
            phase_probabilities[name] = probabilities
            phase_predictions[name] = prediction.tolist()
            phase_gold_nll[name] = -float(gold_log_probabilities.mean())
        residual_predictions = {}
        if len(arms) > 1:
            center = torch.stack(list(phase_probabilities.values())).mean(dim=0)
            for name in arms:
                fused = (
                    (1.0 - args.residual_strength) * phase_probabilities[name]
                    + args.residual_strength * center
                )
                residual_predictions[name] = fused.argmax(dim=-1).tolist()
        base_distribution = phase_probabilities["base0"][0]
        top2 = torch.topk(base_distribution, 2).values
        row = {
            "source_index": sample["source_index"],
            "context_tokens": len(sample["context_ids"]),
            "target_ids": sample["target_ids"],
            "target_text": sample["target_text"],
            "target_tokens": len(sample["target_ids"]),
            "phase_predictions": phase_predictions,
            "phase_gold_nll": phase_gold_nll,
            "residual_predictions": residual_predictions,
            "base_first_token_margin": float(top2[0] - top2[1]),
        }
        rows.append(row)
        print(index + 1, sample["source_index"], {
            "raw": {name: exact(value, sample["target_ids"]) for name, value in phase_predictions.items()},
            "residual": {
                name: exact(value, sample["target_ids"])
                for name, value in residual_predictions.items()
            },
            "target": sample["target_text"],
        }, flush=True)

    summary = summarize(rows, list(arms), args.residual_strength)
    summary.update({
        "dataset_size": dataset_size,
        "excluded_examples": len(excluded),
        "elapsed_seconds": time.perf_counter() - started,
        "examples_per_second": len(rows) / (time.perf_counter() - started),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
    })
    payload = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "summary": summary,
        "rows": rows,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
