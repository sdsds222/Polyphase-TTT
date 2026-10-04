from __future__ import annotations

import json
import math
from pathlib import Path

from transformers import AutoTokenizer

from qa_data import load_hotpot, load_squad
from polyphase_confidence_gate_analysis import evaluate
from ttt_polyphase_real_binding import build_groups, build_hotpot_groups


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent / "results"
KINDS = ("proper", "number", "other")


def payload(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def add_ids(rows, groups):
    ids = [source_id for group in groups for source_id in group["source_ids"]]
    if len(rows) != len(ids):
        raise RuntimeError(f"row/id mismatch: {len(rows)} != {len(ids)}")
    return [dict(row, source_id=source_id) for row, source_id in zip(rows, ids)]


def unique(rows, excluded=None):
    seen = set() if excluded is None else set(excluded)
    kept = []
    for row in rows:
        if row["source_id"] in seen:
            continue
        seen.add(row["source_id"])
        kept.append(row)
    return kept, seen


def paired(rows):
    prediction_removed = 0
    prediction_introduced = 0
    correctness_removed = 0
    correctness_introduced = 0
    default_wins = 0
    default_losses = 0
    for row in rows:
        raw = [int(row["phase_predictions"][name]) for name in ("base0", "base4", "base8", "base12")]
        residual = [int(row["residual_predictions"][name]) for name in ("base0", "base4", "base8", "base12")]
        gold = int(row["gold"])
        raw_flip = len(set(raw)) > 1
        residual_flip = len(set(residual)) > 1
        raw_correctness = len({value == gold for value in raw}) > 1
        residual_correctness = len({value == gold for value in residual}) > 1
        prediction_removed += raw_flip and not residual_flip
        prediction_introduced += not raw_flip and residual_flip
        correctness_removed += raw_correctness and not residual_correctness
        correctness_introduced += not raw_correctness and residual_correctness
        default_wins += raw[0] != gold and residual[0] == gold
        default_losses += raw[0] == gold and residual[0] != gold
    return {
        "prediction_removed": prediction_removed,
        "prediction_introduced": prediction_introduced,
        "correctness_removed": correctness_removed,
        "correctness_introduced": correctness_introduced,
        "default_wins": default_wins,
        "default_losses": default_losses,
    }


def exact_two_sided(removed, introduced):
    discordant = removed + introduced
    if discordant == 0:
        return 1.0
    tail = min(removed, introduced)
    probability = sum(math.comb(discordant, value) for value in range(tail + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * probability)


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "third_party/ttt_mlp_760m_hf",
        trust_remote_code=True,
        local_files_only=True,
    )
    contexts, questions = load_squad(ROOT / "data/squad/dev-v1.1.json")
    squad_rows = {}
    for seed, name in (
        (20261004, "ttt_polyphase_real_binding_350m_broad_n96.json"),
        (20261005, "ttt_polyphase_real_binding_350m_broad_seed2_n96.json"),
        (20261006, "ttt_polyphase_real_binding_350m_broad_seed3_n96.json"),
        (20261008, "ttt_polyphase_online_350m_seed4_n96.json"),
    ):
        groups = build_groups(
            tokenizer,
            contexts,
            questions,
            24,
            512,
            seed,
            answer_kinds=KINDS,
            groups_per_kind=8,
        )
        squad_rows[seed] = add_ids(payload(name)["rows"], groups)
    development, development_ids = unique(squad_rows[20261004] + squad_rows[20261005])
    heldout, _ = unique(squad_rows[20261006] + squad_rows[20261008], development_ids)

    candidates = []
    for step in range(101):
        threshold = step / 100
        result = evaluate(development, threshold)
        full = result["full"]
        if (
            result["default_accuracy"] >= full["default_accuracy"]
            and result["worst_accuracy"] >= full["worst_accuracy"]
            and result["prediction_flip_rate"] <= full["prediction_flip_rate"]
            and result["correctness_flip_rate"] <= full["correctness_flip_rate"]
        ):
            candidates.append((result["mean_phase_count"], threshold, result))
    _, threshold, development_result = min(candidates)

    hotpot_data = load_hotpot(ROOT / "data/hotpotqa/hotpot_dev_distractor_v1.json")
    hotpot_sets = {}
    for seed, groups_per_kind, name in (
        (20261009, 2, "ttt_polyphase_hotpot_760m_screen_n24.json"),
        (20261010, 6, "ttt_polyphase_hotpot_760m_external_n72.json"),
    ):
        groups = build_hotpot_groups(
            tokenizer,
            hotpot_data,
            3 * groups_per_kind,
            1024,
            seed,
            answer_kinds=KINDS,
            groups_per_kind=groups_per_kind,
        )
        hotpot_sets[seed] = add_ids(payload(name)["rows"], groups)
    screen_ids = {row["source_id"] for row in hotpot_sets[20261009]}
    external_hotpot, external_ids = unique(hotpot_sets[20261010], screen_ids)
    expanded_rows = payload("ttt_polyphase_hotpot_760m_external_expand_n144.json")["rows"]
    expanded_hotpot, _ = unique(expanded_rows, external_ids)
    combined_hotpot = external_hotpot + expanded_hotpot

    heldout_result = evaluate(heldout, threshold)
    hotpot_result = evaluate(combined_hotpot, threshold)
    squad_pairs = paired(heldout)
    hotpot_pairs = paired(combined_hotpot)
    combined = {
        key: squad_pairs[key] + hotpot_pairs[key]
        for key in squad_pairs
    }
    combined["prediction_exact_p"] = exact_two_sided(
        combined["prediction_removed"], combined["prediction_introduced"]
    )
    combined["correctness_exact_p"] = exact_two_sided(
        combined["correctness_removed"], combined["correctness_introduced"]
    )
    result = {
        "threshold": threshold,
        "development_squad": {"questions": len(development), "metrics": development_result},
        "heldout_squad": {"questions": len(heldout), "metrics": heldout_result, "paired": squad_pairs},
        "external_hotpotqa": {
            "questions": len(combined_hotpot),
            "original_questions": len(external_hotpot),
            "new_questions": len(expanded_hotpot),
            "metrics": hotpot_result,
            "paired": hotpot_pairs,
        },
        "combined_paired": combined,
    }
    output = RESULTS / "polyphase_disjoint_final_analysis.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("saved", output)


if __name__ == "__main__":
    main()
