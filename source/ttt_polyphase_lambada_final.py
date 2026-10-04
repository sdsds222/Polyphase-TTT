from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ttt_ckpt_util import load_ttt
from ttt_polyphase_lambada import exact, load_examples, mean, paired_sign_p, target_distribution


ROOT = Path(__file__).resolve().parent
PHASES = ("base0", "base4", "base8", "base12")


def confidence(probabilities):
    top = torch.topk(probabilities[0], 2).values
    return float(top[0] - top[1])


def center(values):
    return torch.stack(list(values)).mean(dim=0)


def fuse(anchor, average, strength):
    return (1.0 - strength) * anchor + strength * average


def prediction(probabilities):
    return probabilities.argmax(dim=-1).tolist()


def gold_nll(probabilities, target_ids):
    targets = torch.tensor(target_ids, dtype=torch.long, device=probabilities.device)
    values = probabilities.gather(-1, targets[:, None]).squeeze(-1)
    return -float(values.clamp_min(1e-30).log().mean())


def select_examples(args, tokenizer):
    excluded = set()
    if args.exclude_runs:
        for specification in args.exclude_runs.split(","):
            seed_text, count_text = specification.strip().split(":", 1)
            prior, _ = load_examples(
                args.data,
                tokenizer,
                int(count_text),
                int(seed_text),
                args.max_tokens,
                excluded=excluded,
            )
            excluded.update(row["source_index"] for row in prior)
    selected, dataset_size = load_examples(
        args.data,
        tokenizer,
        args.questions,
        args.seed,
        args.max_tokens,
        excluded=excluded,
    )
    return selected, dataset_size, excluded


def make_arms(model):
    heads = model.config.num_attention_heads
    return {
        "base0": None,
        "base4": {"offsets": [4] * heads},
        "base8": {"offsets": [8] * heads},
        "base12": {"offsets": [12] * heads},
    }


def gated_output(probabilities, anchor_index, threshold, strength):
    anchor = probabilities[anchor_index]
    anchor_prediction = prediction(anchor)
    if confidence(anchor) >= threshold:
        return anchor_prediction, anchor, 1
    opposite_index = (anchor_index + 2) % 4
    opposite = probabilities[opposite_index]
    if anchor_prediction == prediction(opposite):
        output = fuse(anchor, center((anchor, opposite)), strength)
        return prediction(output), output, 2
    output = fuse(anchor, center(probabilities), strength)
    return prediction(output), output, 4


def reset_measurement():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    return time.perf_counter()


def finish_measurement(started, examples):
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "elapsed_seconds": elapsed,
        "examples_per_second": examples / elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
    }


@torch.inference_mode()
def run_baseline(model, selected, settings, label):
    outputs = []
    started = reset_measurement()
    for sample_index, sample in enumerate(selected):
        probabilities, _, _ = target_distribution(
            model, sample["context_ids"], sample["target_ids"], settings
        )
        outputs.append(prediction(probabilities))
        if (sample_index + 1) % 32 == 0 or sample_index + 1 == len(selected):
            print(label, sample_index + 1, "/", len(selected), flush=True)
    metrics = finish_measurement(started, len(selected))
    metrics["predictions"] = outputs
    metrics["accuracy"] = mean([
        exact(value, sample["target_ids"])
        for value, sample in zip(outputs, selected)
    ])
    metrics["mean_phase_count"] = 1.0
    return metrics


@torch.inference_mode()
def run_gated(model, selected, arms, threshold, strength):
    outputs = []
    counts = []
    started = reset_measurement()
    for sample_index, sample in enumerate(selected):
        values = []
        probabilities, _, _ = target_distribution(
            model, sample["context_ids"], sample["target_ids"], arms["base0"]
        )
        values.append(probabilities)
        if confidence(probabilities) >= threshold:
            output = probabilities
            count = 1
        else:
            opposite, _, _ = target_distribution(
                model, sample["context_ids"], sample["target_ids"], arms["base8"]
            )
            values.append(opposite)
            if prediction(probabilities) == prediction(opposite):
                output = fuse(probabilities, center(values), strength)
                count = 2
            else:
                plus, _, _ = target_distribution(
                    model, sample["context_ids"], sample["target_ids"], arms["base4"]
                )
                minus, _, _ = target_distribution(
                    model, sample["context_ids"], sample["target_ids"], arms["base12"]
                )
                output = fuse(probabilities, center((probabilities, plus, opposite, minus)), strength)
                count = 4
        outputs.append(prediction(output))
        counts.append(count)
        if (sample_index + 1) % 32 == 0 or sample_index + 1 == len(selected):
            print("gated", sample_index + 1, "/", len(selected), flush=True)
    metrics = finish_measurement(started, len(selected))
    metrics["predictions"] = outputs
    metrics["accuracy"] = mean([
        exact(value, sample["target_ids"])
        for value, sample in zip(outputs, selected)
    ])
    metrics["mean_phase_count"] = mean(counts)
    metrics["phase_count_histogram"] = {
        str(value): counts.count(value) for value in (1, 2, 4)
    }
    return metrics


@torch.inference_mode()
def run_full(model, selected, arms, threshold, strength):
    rows = []
    started = reset_measurement()
    for sample_index, sample in enumerate(selected):
        phase_probabilities = []
        for name in PHASES:
            probabilities, _, _ = target_distribution(
                model, sample["context_ids"], sample["target_ids"], arms[name]
            )
            phase_probabilities.append(probabilities)
        average = center(phase_probabilities)
        raw = {}
        residual = {}
        gated = {}
        raw_nll = {}
        residual_nll = {}
        gated_nll = {}
        gated_counts = {}
        for anchor_index, name in enumerate(PHASES):
            raw[name] = prediction(phase_probabilities[anchor_index])
            raw_nll[name] = gold_nll(phase_probabilities[anchor_index], sample["target_ids"])
            residual_probabilities = fuse(
                phase_probabilities[anchor_index], average, strength
            )
            residual[name] = prediction(residual_probabilities)
            residual_nll[name] = gold_nll(residual_probabilities, sample["target_ids"])
            gated_prediction, gated_probabilities, count = gated_output(
                phase_probabilities, anchor_index, threshold, strength
            )
            gated[name] = gated_prediction
            gated_nll[name] = gold_nll(gated_probabilities, sample["target_ids"])
            gated_counts[name] = count
        rows.append({
            "source_index": sample["source_index"],
            "target_ids": sample["target_ids"],
            "raw_predictions": raw,
            "residual_predictions": residual,
            "gated_predictions": gated,
            "uniform_prediction": prediction(average),
            "raw_gold_nll": raw_nll,
            "residual_gold_nll": residual_nll,
            "gated_gold_nll": gated_nll,
            "uniform_gold_nll": gold_nll(average, sample["target_ids"]),
            "gated_phase_counts": gated_counts,
        })
        if (sample_index + 1) % 16 == 0 or sample_index + 1 == len(selected):
            print("full", sample_index + 1, "/", len(selected), flush=True)
    metrics = finish_measurement(started, len(selected))
    metrics["rows"] = rows
    return metrics


def method_summary(rows, key, nll_key):
    accuracy = {
        name: mean([
            exact(row[key][name], row["target_ids"]) for row in rows
        ])
        for name in PHASES
    }
    prediction_flips = [
        len({tuple(row[key][name]) for name in PHASES}) > 1 for row in rows
    ]
    correctness_flips = [
        len({exact(row[key][name], row["target_ids"]) for name in PHASES}) > 1
        for row in rows
    ]
    return {
        "phase_accuracy": accuracy,
        "phase_mean_accuracy": mean(list(accuracy.values())),
        "phase_worst_accuracy": min(accuracy.values()),
        "default_accuracy": accuracy["base0"],
        "phase_mean_gold_nll": {
            name: mean([row[nll_key][name] for row in rows]) for name in PHASES
        },
        "prediction_flip_rate": mean(prediction_flips),
        "correctness_flip_rate": mean(correctness_flips),
        "prediction_flips": prediction_flips,
        "correctness_flips": correctness_flips,
    }


def comparison(raw, candidate):
    prediction_removed = sum(
        left and not right
        for left, right in zip(raw["prediction_flips"], candidate["prediction_flips"])
    )
    prediction_introduced = sum(
        not left and right
        for left, right in zip(raw["prediction_flips"], candidate["prediction_flips"])
    )
    correctness_removed = sum(
        left and not right
        for left, right in zip(raw["correctness_flips"], candidate["correctness_flips"])
    )
    correctness_introduced = sum(
        not left and right
        for left, right in zip(raw["correctness_flips"], candidate["correctness_flips"])
    )
    return {
        "prediction_flips_removed": prediction_removed,
        "prediction_flips_introduced": prediction_introduced,
        "prediction_flip_paired_p": paired_sign_p(
            prediction_removed, prediction_introduced
        ),
        "correctness_flips_removed": correctness_removed,
        "correctness_flips_introduced": correctness_introduced,
        "correctness_flip_paired_p": paired_sign_p(
            correctness_removed, correctness_introduced
        ),
    }


def strip_private(summary):
    return {
        key: value for key, value in summary.items()
        if key not in ("prediction_flips", "correctness_flips")
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/lambada/lambada_test.jsonl")
    parser.add_argument("--questions", type=int, default=384)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--exclude-runs", default="20261004:64,20261005:96")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.04)
    parser.add_argument("--residual-strength", type=float, default=0.25)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    model, tokenizer = load_ttt(dtype=torch.float32, model_dir=args.model_dir)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected, dataset_size, excluded = select_examples(args, tokenizer)
    arms = make_arms(model)
    target_distribution(
        model,
        selected[0]["context_ids"],
        selected[0]["target_ids"],
        arms["base0"],
    )

    baseline_pre = run_baseline(model, selected, arms["base0"], "baseline_pre")
    print("baseline_pre", {k: v for k, v in baseline_pre.items() if k != "predictions"}, flush=True)
    gated_timed = run_gated(
        model, selected, arms, args.threshold, args.residual_strength
    )
    print("gated", {k: v for k, v in gated_timed.items() if k != "predictions"}, flush=True)
    full = run_full(model, selected, arms, args.threshold, args.residual_strength)
    print("full", {k: v for k, v in full.items() if k != "rows"}, flush=True)
    baseline_post = run_baseline(model, selected, arms["base0"], "baseline_post")
    print("baseline_post", {k: v for k, v in baseline_post.items() if k != "predictions"}, flush=True)

    baseline_seconds = mean([
        baseline_pre["elapsed_seconds"], baseline_post["elapsed_seconds"]
    ])
    rows = full.pop("rows")
    raw = method_summary(rows, "raw_predictions", "raw_gold_nll")
    residual = method_summary(rows, "residual_predictions", "residual_gold_nll")
    gated = method_summary(rows, "gated_predictions", "gated_gold_nll")
    uniform_accuracy = mean([
        exact(row["uniform_prediction"], row["target_ids"]) for row in rows
    ])
    uniform_nll = mean([row["uniform_gold_nll"] for row in rows])
    offline_gated_base0 = [row["gated_predictions"]["base0"] for row in rows]
    full_residual_base0 = [row["residual_predictions"]["base0"] for row in rows]
    source_indices = [row["source_index"] for row in rows]

    timing = {
        "baseline_seconds": baseline_seconds,
        "baseline_pre_seconds": baseline_pre["elapsed_seconds"],
        "baseline_post_seconds": baseline_post["elapsed_seconds"],
        "gated_seconds": gated_timed["elapsed_seconds"],
        "full_seconds": full["elapsed_seconds"],
        "gated_latency_multiple": gated_timed["elapsed_seconds"] / baseline_seconds,
        "full_latency_multiple": full["elapsed_seconds"] / baseline_seconds,
        "gated_mean_phase_count": gated_timed["mean_phase_count"],
        "gated_phase_count_histogram": gated_timed["phase_count_histogram"],
        "gated_prediction_match_offline": mean([
            left == right
            for left, right in zip(gated_timed["predictions"], offline_gated_base0)
        ]),
        "gated_prediction_match_full_residual": mean([
            left == right
            for left, right in zip(gated_timed["predictions"], full_residual_base0)
        ]),
        "baseline_peak_allocated_gib": max(
            baseline_pre["peak_allocated_gib"], baseline_post["peak_allocated_gib"]
        ),
        "gated_peak_allocated_gib": gated_timed["peak_allocated_gib"],
        "full_peak_allocated_gib": full["peak_allocated_gib"],
    }
    summary = {
        "examples": len(rows),
        "dataset_size": dataset_size,
        "excluded_examples": len(excluded),
        "source_indices": source_indices,
        "raw": strip_private(raw),
        "uniform": {
            "accuracy": uniform_accuracy,
            "gold_nll": uniform_nll,
            "prediction_flip_rate": 0.0,
            "correctness_flip_rate": 0.0,
        },
        "anchored_residual": strip_private(residual),
        "confidence_cascade": strip_private(gated),
        "anchored_residual_vs_raw": comparison(raw, residual),
        "confidence_cascade_vs_raw": comparison(raw, gated),
        "timing": timing,
    }
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
