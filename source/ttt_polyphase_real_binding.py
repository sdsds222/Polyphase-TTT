from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from qa_data import load_hotpot, load_squad
from ttt_ckpt_util import load_ttt
from ttt_staggered_chunk_screen import staggered_calendar


ROOT = Path(__file__).resolve().parent
DIRECTED_KERNEL = (0.70, 0.25, 0.0, 0.05)

NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand", "million", "billion", "first", "second", "third", "fourth",
    "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "half",
    "quarter", "dozen",
}


def prompt_text(document, question):
    return (
        "Read the passages and answer the question using only the passages.\n\n"
        + document
        + "\n\nQuestion: "
        + question
        + "\nAnswer:"
    )


def hotpot_prompt_text(document, question):
    return (
        "Read the evidence bundles and answer the multi-hop question using only the evidence.\n\n"
        + document
        + "\n\nQuestion: "
        + question
        + "\nAnswer:"
    )


def answer_kind(answer):
    normalized = answer.strip().lower().strip(".,;:!?()[]{}")
    if any(character.isdigit() for character in answer) or normalized in NUMBER_WORDS:
        return "number"
    if answer[:1].isupper():
        return "proper"
    return "other"


def build_groups(
    tokenizer,
    contexts,
    questions,
    count,
    length,
    seed,
    answer_kinds=None,
    groups_per_kind=0,
    exclude_source_ids=None,
    mixed_kinds=False,
):
    rng = random.Random(seed)
    excluded = set() if exclude_source_ids is None else set(exclude_source_ids)
    context_lengths = {}
    eligible = []
    for context_id, context, question, answers in questions:
        if not answers:
            continue
        answer = answers[0]
        source_id = f"{context_id}\t{question}"
        if source_id in excluded:
            continue
        answer_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
        if len(answer_ids) != 1:
            continue
        if context_id not in context_lengths:
            context_lengths[context_id] = len(tokenizer.encode(context, add_special_tokens=False))
        if not 24 <= context_lengths[context_id] <= 150:
            continue
        kind = answer_kind(answer)
        eligible.append((context_id, context, question, answer, answer_ids[0], kind, source_id))
    buckets = {}
    for item in eligible:
        buckets.setdefault(item[5], []).append(item)
    if mixed_kinds:
        allowed = set(answer_kinds) if answer_kinds else set(buckets)
        buckets = {"mixed": [item for item in eligible if item[5] in allowed]}
    legacy_selection = not answer_kinds and not groups_per_kind
    if answer_kinds and not mixed_kinds:
        kinds = list(answer_kinds)
        missing = [kind for kind in kinds if kind not in buckets]
        if missing:
            raise RuntimeError(f"no eligible examples for answer kinds: {missing}")
    else:
        kinds = list(buckets)
        rng.shuffle(kinds)
    for kind in kinds:
        rng.shuffle(buckets[kind])
    if legacy_selection:
        quotas = {kind: count for kind in kinds}
    elif groups_per_kind:
        quotas = {kind: groups_per_kind for kind in kinds}
        count = groups_per_kind * len(kinds)
    else:
        quotas = {}
        remaining = count
        for index, kind in enumerate(kinds):
            slots = len(kinds) - index
            quota = math.ceil(remaining / slots)
            quotas[kind] = quota
            remaining -= quota
    groups = []
    for kind in kinds:
        available = list(buckets[kind])
        built = 0
        while (
            len(available) >= 4
            and built < quotas[kind]
            and len(groups) < count
        ):
            chosen = None
            for _ in range(2000):
                indices = rng.sample(range(len(available)), 4)
                trial = [available[index] for index in indices]
                if len({item[0] for item in trial}) < 4:
                    continue
                if len({item[4] for item in trial}) < 4:
                    continue
                document = "\n\n".join(
                    f"[Passage {index + 1}]\n{value[1]}"
                    for index, value in enumerate(trial)
                )
                if max(
                    len(tokenizer.encode(prompt_text(document, value[2]), add_special_tokens=False))
                    for value in trial
                ) > length:
                    continue
                chosen = indices
                break
            if chosen is None:
                break
            selected = [available[index] for index in chosen]
            for index in sorted(chosen, reverse=True):
                available.pop(index)
            rng.shuffle(selected)
            document = "\n\n".join(
                f"[Passage {index + 1}]\n{item[1]}" for index, item in enumerate(selected)
            )
            prompts = [prompt_text(document, item[2]) for item in selected]
            groups.append({
                "prompts": prompts,
                "answers": [item[3] for item in selected],
                "answer_ids": [item[4] for item in selected],
                "answer_kind": "mixed" if mixed_kinds else selected[0][5],
                "source_ids": [item[6] for item in selected],
            })
            built += 1
        if not legacy_selection and built < quotas[kind]:
            raise RuntimeError(
                f"constructed only {built} of {quotas[kind]} groups for {kind}"
            )
        if len(groups) >= count:
            break
    if len(groups) < count:
        raise RuntimeError(f"constructed only {len(groups)} of {count} groups")
    rng.shuffle(groups)
    return groups


def build_hotpot_groups(
    tokenizer,
    data,
    count,
    length,
    seed,
    answer_kinds=None,
    groups_per_kind=0,
    exclude_source_ids=None,
    mixed_kinds=False,
):
    rng = random.Random(seed)
    excluded = set() if exclude_source_ids is None else set(exclude_source_ids)
    eligible = []
    for item in data:
        if item["_id"] in excluded:
            continue
        answer = item["answer"].strip()
        if answer.lower() in {"yes", "no"}:
            continue
        answer_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
        if len(answer_ids) != 1:
            continue
        support_titles = {title for title, _ in item["supporting_facts"]}
        passages = []
        for title, sentences in item["context"]:
            if title in support_titles:
                passages.append(f"[{title}]\n{''.join(sentences).strip()}")
        if len(passages) < 2:
            continue
        evidence = "\n\n".join(passages)
        evidence_length = len(tokenizer.encode(evidence, add_special_tokens=False))
        if not 24 <= evidence_length <= 500:
            continue
        eligible.append({
            "id": item["_id"],
            "evidence": evidence,
            "question": item["question"].strip(),
            "answer": answer,
            "answer_id": answer_ids[0],
            "kind": answer_kind(answer),
        })
    buckets = {}
    for item in eligible:
        buckets.setdefault(item["kind"], []).append(item)
    if mixed_kinds:
        allowed = set(answer_kinds) if answer_kinds else set(buckets)
        buckets = {"mixed": [item for item in eligible if item["kind"] in allowed]}
    if answer_kinds and not mixed_kinds:
        kinds = list(answer_kinds)
        missing = [kind for kind in kinds if kind not in buckets]
        if missing:
            raise RuntimeError(f"no eligible HotpotQA examples for answer kinds: {missing}")
    else:
        kinds = list(buckets)
        rng.shuffle(kinds)
    for kind in kinds:
        rng.shuffle(buckets[kind])
    if groups_per_kind:
        quotas = {kind: groups_per_kind for kind in kinds}
        count = groups_per_kind * len(kinds)
    else:
        quotas = {}
        remaining = count
        for index, kind in enumerate(kinds):
            slots = len(kinds) - index
            quota = math.ceil(remaining / slots)
            quotas[kind] = quota
            remaining -= quota
    groups = []
    for kind in kinds:
        available = list(buckets[kind])
        built = 0
        while len(available) >= 4 and built < quotas[kind] and len(groups) < count:
            chosen = None
            for _ in range(4000):
                indices = rng.sample(range(len(available)), 4)
                trial = [available[index] for index in indices]
                if len({item["answer_id"] for item in trial}) < 4:
                    continue
                document = "\n\n".join(
                    f"[Evidence bundle {index + 1}]\n{value['evidence']}"
                    for index, value in enumerate(trial)
                )
                if max(
                    len(tokenizer.encode(
                        hotpot_prompt_text(document, value["question"]),
                        add_special_tokens=False,
                    ))
                    for value in trial
                ) > length:
                    continue
                chosen = indices
                break
            if chosen is None:
                break
            selected = [available[index] for index in chosen]
            for index in sorted(chosen, reverse=True):
                available.pop(index)
            rng.shuffle(selected)
            document = "\n\n".join(
                f"[Evidence bundle {index + 1}]\n{item['evidence']}"
                for index, item in enumerate(selected)
            )
            groups.append({
                "prompts": [
                    hotpot_prompt_text(document, item["question"])
                    for item in selected
                ],
                "answers": [item["answer"] for item in selected],
                "answer_ids": [item["answer_id"] for item in selected],
                "answer_kind": "mixed" if mixed_kinds else selected[0]["kind"],
                "source_ids": [item["id"] for item in selected],
            })
            built += 1
        if built < quotas[kind]:
            raise RuntimeError(
                f"constructed only {built} of {quotas[kind]} HotpotQA groups for {kind}"
            )
    if len(groups) < count:
        raise RuntimeError(f"constructed only {len(groups)} of {count} HotpotQA groups")
    rng.shuffle(groups)
    return groups


@torch.inference_mode()
def losses_for_ids(model, ids, answer_ids, settings, return_prompt_losses=False):
    if settings is None:
        logits = model(input_ids=ids, use_cache=False, return_dict=True).logits[0].float()
    else:
        with staggered_calendar(model, **settings):
            logits = model(input_ids=ids, use_cache=False, return_dict=True).logits[0].float()
    answer_log_probs = logits[-1].log_softmax(dim=-1)
    answer_losses = [-float(answer_log_probs[index]) for index in answer_ids]
    if not return_prompt_losses:
        return answer_losses
    prompt_losses = F.cross_entropy(
        logits[:-1], ids[0, 1:], reduction="none"
    ).detach().cpu()
    return answer_losses, prompt_losses


def online_expert_weights(loss_matrix, half_life=64.0, eta=1.0, floor=0.05):
    decay = 0.5 ** (1.0 / half_life)
    log_weights = torch.zeros(loss_matrix.shape[0], dtype=torch.float64)
    for token_losses in loss_matrix.transpose(0, 1).double():
        relative = (token_losses - token_losses.mean()).clamp(-2.0, 2.0)
        log_weights = decay * log_weights - eta * relative
    weights = torch.softmax(log_weights, dim=0)
    return floor + (1.0 - floor * len(weights)) * weights


def prediction(losses):
    return min(range(len(losses)), key=losses.__getitem__)


def mean(values):
    return sum(values) / len(values)


def phase_summary(rows, phase_names):
    phase_accuracies = {
        name: mean([row["phase_predictions"][name] == row["gold"] for row in rows])
        for name in phase_names
    }
    residual_accuracies = {
        name: mean([row["residual_predictions"][name] == row["gold"] for row in rows])
        for name in phase_names
    }
    flip = [len(set(row["phase_predictions"].values())) > 1 for row in rows]
    correctness_flip = [
        len({row["phase_predictions"][name] == row["gold"] for name in phase_names}) > 1
        for row in rows
    ]
    residual_flip = [
        len(set(row["residual_predictions"].values())) > 1 for row in rows
    ]
    residual_correctness_flip = [
        len({row["residual_predictions"][name] == row["gold"] for name in phase_names}) > 1
        for row in rows
    ]
    summary = {
        "questions": len(rows),
        "phase_accuracies": phase_accuracies,
        "phase_mean_accuracy": mean(list(phase_accuracies.values())),
        "phase_worst_accuracy": min(phase_accuracies.values()),
        "phase_best_accuracy": max(phase_accuracies.values()),
        "prediction_flip_rate": mean(flip),
        "correctness_flip_rate": mean(correctness_flip),
        "residual_phase_accuracies": residual_accuracies,
        "residual_default_accuracy": residual_accuracies["base0"],
        "residual_worst_accuracy": min(residual_accuracies.values()),
        "residual_best_accuracy": max(residual_accuracies.values()),
        "residual_accuracy_span": max(residual_accuracies.values()) - min(residual_accuracies.values()),
        "residual_prediction_flip_rate": mean(residual_flip),
        "residual_correctness_flip_rate": mean(residual_correctness_flip),
        "poly4_accuracy": mean([row["poly4_prediction"] == row["gold"] for row in rows]),
        "adaptive_accuracy": mean([row["adaptive_prediction"] == row["gold"] for row in rows]),
        "adaptive_expanded_rate": mean([row["expanded"] for row in rows]),
        "adaptive_mean_phase_count": 2.0 + 2.0 * mean([row["expanded"] for row in rows]),
        "adaptive_matches_poly4": mean([
            row["adaptive_prediction"] == row["poly4_prediction"] for row in rows
        ]),
    }
    if all("directed_predictions" in row for row in rows):
        directed_accuracies = {
            name: mean([row["directed_predictions"][name] == row["gold"] for row in rows])
            for name in phase_names
        }
        directed_flip = [
            len(set(row["directed_predictions"].values())) > 1 for row in rows
        ]
        directed_correctness_flip = [
            len({row["directed_predictions"][name] == row["gold"] for name in phase_names}) > 1
            for row in rows
        ]
        summary.update({
            "directed_kernel": list(DIRECTED_KERNEL),
            "directed_phase_accuracies": directed_accuracies,
            "directed_default_accuracy": directed_accuracies["base0"],
            "directed_worst_accuracy": min(directed_accuracies.values()),
            "directed_best_accuracy": max(directed_accuracies.values()),
            "directed_prediction_flip_rate": mean(directed_flip),
            "directed_correctness_flip_rate": mean(directed_correctness_flip),
        })
    if all("online_predictions" in row for row in rows):
        online_accuracies = {
            name: mean([row["online_predictions"][name] == row["gold"] for row in rows])
            for name in phase_names
        }
        online_flip = [
            len(set(row["online_predictions"].values())) > 1 for row in rows
        ]
        online_correctness_flip = [
            len({row["online_predictions"][name] == row["gold"] for name in phase_names}) > 1
            for row in rows
        ]
        summary.update({
            "online_phase_accuracies": online_accuracies,
            "online_default_accuracy": online_accuracies["base0"],
            "online_worst_accuracy": min(online_accuracies.values()),
            "online_best_accuracy": max(online_accuracies.values()),
            "online_prediction_flip_rate": mean(online_flip),
            "online_correctness_flip_rate": mean(online_correctness_flip),
            "online_mean_max_weight": mean([max(row["online_weights"]) for row in rows]),
        })
    return summary


def boundary_bucket(distance):
    if distance is None:
        return "missing"
    if distance <= 1:
        return "edge_0_1"
    if distance <= 3:
        return "near_2_3"
    return "interior_4_7"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=ROOT / "third_party/ttt_mlp_125m")
    ap.add_argument("--dataset", choices=("squad", "hotpotqa"), default="squad")
    ap.add_argument("--data", type=Path, default=ROOT / "data/squad/dev-v1.1.json")
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--answer-kinds", default="")
    ap.add_argument("--groups-per-kind", type=int, default=0)
    ap.add_argument("--exclude-seeds", default="")
    ap.add_argument("--exclude-groups-per-kind", type=int, default=0)
    ap.add_argument("--exclude-specs", default="")
    ap.add_argument("--mixed-answer-kinds", action="store_true")
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--residual-strength", type=float, default=0.25)
    ap.add_argument("--online-half-life", type=float, default=64.0)
    ap.add_argument("--online-eta", type=float, default=1.0)
    ap.add_argument("--online-floor", type=float, default=0.05)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "ttt_polyphase_real_binding.json")
    args = ap.parse_args()

    model, tokenizer = load_ttt(dtype=torch.float32, model_dir=args.model_dir)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    answer_kinds = [value.strip() for value in args.answer_kinds.split(",") if value.strip()]
    exclude_seeds = [int(value) for value in args.exclude_seeds.split(",") if value.strip()]
    exclude_specs = []
    for value in args.exclude_specs.split(","):
        value = value.strip()
        if not value:
            continue
        fields = value.split(":")
        if len(fields) not in (2, 3):
            raise ValueError("exclude specs must be seed:groups or seed:groups:length")
        seed = int(fields[0])
        groups_per_kind = int(fields[1])
        length = int(fields[2]) if len(fields) == 3 else args.length
        exclude_specs.append((seed, groups_per_kind, length))
    if not exclude_specs:
        exclude_specs = [
            (seed, args.exclude_groups_per_kind, args.length)
            for seed in exclude_seeds
        ]
    excluded = set()
    if args.dataset == "hotpotqa":
        data_path = args.data
        if data_path == ROOT / "data/squad/dev-v1.1.json":
            data_path = ROOT / "data/hotpotqa/hotpot_dev_distractor_v1.json"
        hotpot_data = load_hotpot(data_path)
        for seed, groups_per_kind, prior_length in exclude_specs:
            prior = build_hotpot_groups(
                tokenizer,
                hotpot_data,
                len(answer_kinds) * groups_per_kind,
                prior_length,
                seed,
                answer_kinds=answer_kinds,
                groups_per_kind=groups_per_kind,
            )
            excluded.update(source_id for group in prior for source_id in group["source_ids"])
        groups = build_hotpot_groups(
            tokenizer,
            hotpot_data,
            args.groups,
            args.length,
            args.seed,
            answer_kinds=answer_kinds,
            groups_per_kind=args.groups_per_kind,
            exclude_source_ids=excluded,
            mixed_kinds=args.mixed_answer_kinds,
        )
    else:
        contexts, questions = load_squad(args.data)
        for seed, groups_per_kind, prior_length in exclude_specs:
            prior = build_groups(
                tokenizer,
                contexts,
                questions,
                len(answer_kinds) * groups_per_kind,
                prior_length,
                seed,
                answer_kinds=answer_kinds,
                groups_per_kind=groups_per_kind,
            )
            excluded.update(source_id for group in prior for source_id in group["source_ids"])
        groups = build_groups(
            tokenizer,
            contexts,
            questions,
            args.groups,
            args.length,
            args.seed,
            answer_kinds=answer_kinds,
            groups_per_kind=args.groups_per_kind,
            exclude_source_ids=excluded,
            mixed_kinds=args.mixed_answer_kinds,
        )
    if args.build_only:
        print(json.dumps({
            "groups": len(groups),
            "questions": 4 * len(groups),
            "answer_kinds": [group["answer_kind"] for group in groups],
        }, indent=2))
        return
    heads = model.config.num_attention_heads
    arms = {
        "base0": None,
        "base4": {"offsets": [4] * heads},
        "base8": {"offsets": [8] * heads},
        "base12": {"offsets": [12] * heads},
    }
    rows = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for group_index, group in enumerate(groups):
        for query_index, prompt in enumerate(group["prompts"]):
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            ids = torch.tensor([prompt_ids], device="cuda")
            phase_losses = {}
            phase_prompt_losses = {}
            phase_predictions = {}
            for name, settings in arms.items():
                losses, prompt_losses = losses_for_ids(
                    model, ids, group["answer_ids"], settings, return_prompt_losses=True
                )
                phase_losses[name] = losses
                phase_prompt_losses[name] = prompt_losses
                phase_predictions[name] = prediction(losses)
            matrix = torch.tensor([phase_losses[name] for name in arms], dtype=torch.float64)
            phase_probabilities = torch.exp(-matrix)
            mean_probabilities = phase_probabilities.mean(dim=0)
            online_weights = online_expert_weights(
                torch.stack([phase_prompt_losses[name] for name in arms]),
                half_life=args.online_half_life,
                eta=args.online_eta,
                floor=args.online_floor,
            )
            online_center = sum(
                online_weights[index] * phase_probabilities[index]
                for index in range(len(arms))
            )
            poly4_losses = (-torch.logsumexp(-matrix, dim=0) + math.log(4.0)).tolist()
            poly2_losses = (-torch.logsumexp(-matrix[[0, 2]], dim=0) + math.log(2.0)).tolist()
            expand = int(phase_predictions["base0"] != phase_predictions["base8"])
            adaptive_losses = poly4_losses if expand else poly2_losses
            residual_predictions = {}
            directed_predictions = {}
            online_predictions = {}
            for phase_index, name in enumerate(arms):
                fused = (
                    (1.0 - args.residual_strength) * phase_probabilities[phase_index]
                    + args.residual_strength * mean_probabilities
                )
                residual_predictions[name] = int(fused.argmax())
                directed = sum(
                    DIRECTED_KERNEL[relative] * phase_probabilities[(phase_index + relative) % 4]
                    for relative in range(4)
                )
                directed_predictions[name] = int(directed.argmax())
                online = (
                    (1.0 - args.residual_strength) * phase_probabilities[phase_index]
                    + args.residual_strength * online_center
                )
                online_predictions[name] = int(online.argmax())
            row = {
                "group": group_index,
                "query": query_index,
                "answer_kind": group["answer_kind"],
                "source_id": group["source_ids"][query_index],
                "prompt_tokens": len(prompt_ids),
                "phase_losses": phase_losses,
                "phase_predictions": phase_predictions,
                "poly4_prediction": prediction(poly4_losses),
                "adaptive_prediction": prediction(adaptive_losses),
                "residual_predictions": residual_predictions,
                "directed_predictions": directed_predictions,
                "online_predictions": online_predictions,
                "online_weights": online_weights.tolist(),
                "expanded": expand,
                "gold": query_index,
            }
            gold_id = group["answer_ids"][query_index]
            gold_positions = [
                index for index, token_id in enumerate(prompt_ids) if token_id == gold_id
            ]
            row["gold_token_positions"] = gold_positions
            row["base_boundary_distance"] = (
                min(min(position % 16, 15 - (position % 16)) for position in gold_positions)
                if gold_positions else None
            )
            rows.append(row)
            print(group_index, query_index, {
                "phase_predictions": list(phase_predictions.values()),
                "poly4": row["poly4_prediction"],
                "adaptive": row["adaptive_prediction"],
                "directed": row["directed_predictions"]["base0"],
                "online": row["online_predictions"]["base0"],
                "gold": query_index,
                "expanded": expand,
            }, flush=True)

    phase_names = list(arms)
    summary = phase_summary(rows, phase_names)
    summary["dataset"] = args.dataset
    summary["residual_strength"] = args.residual_strength
    summary["online_half_life"] = args.online_half_life
    summary["online_eta"] = args.online_eta
    summary["online_floor"] = args.online_floor
    summary["elapsed_seconds"] = time.perf_counter() - started
    summary["questions_per_second"] = len(rows) / summary["elapsed_seconds"]
    summary["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / (1024 ** 3)
    summary["by_answer_kind"] = {
        kind: phase_summary(
            [row for row in rows if row["answer_kind"] == kind], phase_names
        )
        for kind in sorted({row["answer_kind"] for row in rows})
    }
    summary["by_boundary_distance"] = {
        bucket: phase_summary(
            [
                row for row in rows
                if boundary_bucket(row["base_boundary_distance"]) == bucket
            ],
            phase_names,
        )
        for bucket in ("edge_0_1", "near_2_3", "interior_4_7", "missing")
        if any(
            boundary_bucket(row["base_boundary_distance"]) == bucket for row in rows
        )
    }
    payload = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "summary": summary,
        "rows": rows,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("saved", args.out)


if __name__ == "__main__":
    main()
