from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NAMES = ("base0", "base4", "base8", "base12")
RHO = 0.25


def load_rows(names):
    rows = []
    for name in names:
        payload = json.loads((ROOT / name).read_text(encoding="utf-8"))
        rows.extend(payload["rows"])
    return rows


def probabilities(row):
    return [[math.exp(-float(value)) for value in row["phase_losses"][name]] for name in NAMES]


def argmax(values):
    return max(range(len(values)), key=values.__getitem__)


def confidence(values):
    total = sum(values)
    normalized = sorted((value / total for value in values), reverse=True)
    return normalized[0] - normalized[1]


def fuse(anchor, center):
    return [(1.0 - RHO) * left + RHO * right for left, right in zip(anchor, center)]


def average(items):
    return [sum(values) / len(values) for values in zip(*items)]


def evaluate(rows, threshold):
    outputs = []
    phase_counts = []
    full_outputs = []
    raw_outputs = []
    for row in rows:
        probs = probabilities(row)
        raw = [argmax(values) for values in probs]
        center4 = average(probs)
        full = [argmax(fuse(probs[index], center4)) for index in range(4)]
        gated = []
        counts = []
        for index in range(4):
            if confidence(probs[index]) >= threshold:
                gated.append(raw[index])
                counts.append(1)
                continue
            opposite = (index + 2) % 4
            if raw[index] == raw[opposite]:
                center2 = average((probs[index], probs[opposite]))
                gated.append(argmax(fuse(probs[index], center2)))
                counts.append(2)
            else:
                gated.append(full[index])
                counts.append(4)
        outputs.append(gated)
        phase_counts.append(counts)
        full_outputs.append(full)
        raw_outputs.append(raw)
    gold = [int(row["gold"]) for row in rows]

    def metrics(values):
        accuracies = [
            sum(item[index] == target for item, target in zip(values, gold)) / len(gold)
            for index in range(4)
        ]
        flips = sum(len(set(item)) > 1 for item in values) / len(values)
        correctness = sum(
            len({prediction == target for prediction in item}) > 1
            for item, target in zip(values, gold)
        ) / len(values)
        return {
            "default_accuracy": accuracies[0],
            "mean_accuracy": sum(accuracies) / 4,
            "worst_accuracy": min(accuracies),
            "prediction_flip_rate": flips,
            "correctness_flip_rate": correctness,
        }

    result = metrics(outputs)
    result["mean_phase_count"] = sum(map(sum, phase_counts)) / (4 * len(rows))
    result["full_prediction_match"] = sum(
        left == right
        for item, full in zip(outputs, full_outputs)
        for left, right in zip(item, full)
    ) / (4 * len(rows))
    result["raw"] = metrics(raw_outputs)
    result["full"] = metrics(full_outputs)
    return result


def main():
    development = load_rows((
        "ttt_polyphase_real_binding_350m_broad_n96.json",
        "ttt_polyphase_real_binding_350m_broad_seed2_n96.json",
    ))
    candidates = []
    for step in range(101):
        threshold = step / 100
        result = evaluate(development, threshold)
        full = result["full"]
        safe = (
            result["default_accuracy"] >= full["default_accuracy"]
            and result["worst_accuracy"] >= full["worst_accuracy"]
            and result["prediction_flip_rate"] <= full["prediction_flip_rate"]
            and result["correctness_flip_rate"] <= full["correctness_flip_rate"]
        )
        if safe:
            candidates.append((result["mean_phase_count"], threshold, result))
    if not candidates:
        raise RuntimeError("no safe confidence threshold on development data")
    _, threshold, development_result = min(candidates)
    evaluations = {
        "development_squad_350m_n192": development_result,
        "heldout_squad_350m_n192": evaluate(load_rows((
            "ttt_polyphase_real_binding_350m_broad_seed3_n96.json",
            "ttt_polyphase_online_350m_seed4_n96.json",
        )), threshold),
        "external_hotpotqa_760m_n72": evaluate(load_rows((
            "ttt_polyphase_hotpot_760m_external_n72.json",
        )), threshold),
    }
    payload = {
        "threshold": threshold,
        "residual_strength": RHO,
        "selection_rule": "minimum mean phase count subject to no degradation versus full residual on development",
        "evaluations": evaluations,
    }
    output = ROOT / "polyphase_confidence_gate_analysis.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print("saved", output)


if __name__ == "__main__":
    main()
