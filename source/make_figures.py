from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT.parent / "results"
FIGURES = ROOT.parent / "figures"
DATA = ROOT.parent / "figure_data"

INK = "#252A30"
RED = "#B6403A"
DARK_RED = "#87312E"
BLUE = "#477A9E"
DARK_BLUE = "#315872"
GRAY = "#7D858D"
MID_GRAY = "#A9AFB4"
LIGHT_GRAY = "#E4E6E8"
PALE_GRAY = "#F4F5F6"
PALE_RED = "#F7E8E6"
PALE_BLUE = "#E8F0F5"
WHITE = "#FFFFFF"
TEXT_WIDTH = 7.12

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.4,
    "axes.titlesize": 8.4,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 6.6,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
})


def read_json(name):
    return json.loads((EXPERIMENTS / name).read_text(encoding="utf-8"))


def save(fig, stem):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.035, facecolor=WHITE)
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight", pad_inches=0.035, facecolor=WHITE)
    fig.savefig(FIGURES / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.035, facecolor=WHITE)
    plt.close(fig)


def style_axis(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MID_GRAY)
    ax.tick_params(length=2.5, color=MID_GRAY)
    if grid:
        ax.grid(axis=grid, color=LIGHT_GRAY, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def panel(ax, letter, title):
    ax.set_title(f"{letter}  {title}", loc="left", pad=8, color=INK, fontweight="bold")


def wire(ax, points, color=GRAY, lw=1.0, dashed=False, arrow=True, zorder=2):
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if not (math.isclose(x0, x1) or math.isclose(y0, y1)):
            raise ValueError(f"non-orthogonal segment: {(x0, y0)} -> {(x1, y1)}")
    style = (0, (3, 2.2)) if dashed else "solid"
    if len(points) > 2:
        ax.plot([p[0] for p in points[:-1]], [p[1] for p in points[:-1]],
                color=color, linewidth=lw, linestyle=style,
                solid_joinstyle="miter", zorder=zorder)
    if arrow:
        ax.add_patch(FancyArrowPatch(
            points[-2], points[-1], arrowstyle="-|>", mutation_scale=7.5,
            color=color, linewidth=lw, linestyle=style,
            shrinkA=0, shrinkB=0, connectionstyle="arc3", zorder=zorder,
        ))
    else:
        ax.plot([p[0] for p in points[-2:]], [p[1] for p in points[-2:]],
                color=color, linewidth=lw, linestyle=style, zorder=zorder)


def box(ax, xy, wh, title, subtitle="", edge=GRAY, face=WHITE,
        title_color=None, title_size=7.1, subtitle_size=6.2):
    x, y = xy
    w, h = wh
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.016,rounding_size=0.055",
        linewidth=1.0, edgecolor=edge, facecolor=face, zorder=3,
    ))
    ax.text(x + w / 2, y + h * (0.62 if subtitle else 0.5), title,
            ha="center", va="center", color=title_color or edge,
            fontsize=title_size, fontweight="bold", zorder=4)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.28, subtitle,
                ha="center", va="center", color=INK,
                fontsize=subtitle_size, zorder=4)


def load_primary():
    files = {
        "350M": "ttt_polyphase_lambada_final_350m_n384.json",
        "760M": "ttt_polyphase_lambada_final_760m_n384.json",
    }
    return {model: read_json(name) for model, name in files.items()}, files


def prepare_data():
    primary, files = load_primary()
    records = []
    for model, payload in primary.items():
        summary = payload["summary"]
        timing = summary["timing"]
        methods = {
            "Official phase": {
                "metrics": summary["raw"],
                "latency": 1.0,
                "phases": 1.0,
                "memory": timing["baseline_peak_allocated_gib"],
            },
            "Uniform four-phase": {
                "metrics": {
                    "default_accuracy": summary["uniform"]["accuracy"],
                    "phase_mean_accuracy": summary["uniform"]["accuracy"],
                    "phase_worst_accuracy": summary["uniform"]["accuracy"],
                    "prediction_flip_rate": 0.0,
                    "correctness_flip_rate": 0.0,
                },
                "latency": timing["full_latency_multiple"],
                "phases": 4.0,
                "memory": timing["full_peak_allocated_gib"],
            },
            "Anchored residual": {
                "metrics": summary["anchored_residual"],
                "latency": timing["full_latency_multiple"],
                "phases": 4.0,
                "memory": timing["full_peak_allocated_gib"],
            },
            "Confidence cascade": {
                "metrics": summary["confidence_cascade"],
                "latency": timing["gated_latency_multiple"],
                "phases": timing["gated_mean_phase_count"],
                "memory": timing["gated_peak_allocated_gib"],
            },
        }
        baseline_memory = timing["baseline_peak_allocated_gib"]
        for method, item in methods.items():
            metrics = item["metrics"]
            records.append({
                "model": model,
                "examples": summary["examples"],
                "method": method,
                "default_accuracy": metrics["default_accuracy"],
                "mean_phase_accuracy": metrics["phase_mean_accuracy"],
                "worst_phase_accuracy": metrics["phase_worst_accuracy"],
                "prediction_flip_rate": metrics["prediction_flip_rate"],
                "correctness_flip_rate": metrics["correctness_flip_rate"],
                "latency_multiple": item["latency"],
                "mean_phase_count": item["phases"],
                "peak_memory_gib": item["memory"],
                "peak_memory_multiple": item["memory"] / baseline_memory,
            })
    DATA.mkdir(parents=True, exist_ok=True)
    columns = list(records[0])
    with (DATA / "primary_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    source_files = list(files.values()) + [
        "polyphase_disjoint_final_analysis.json",
        "polyphase_fast_benchmark_squad_350m_n24.json",
        "polyphase_fast_benchmark_hotpot_760m_n24.json",
    ]
    manifest = {}
    for name in source_files:
        path = EXPERIMENTS / name
        manifest[name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (DATA / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return primary, records


def figure_architecture():
    fig = plt.figure(figsize=(TEXT_WIDTH, 3.55))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(0.18, 5.72, "(a)  Chunk origin defines TTT's internal calendar",
            fontsize=8.2, fontweight="bold", color=INK, ha="left", va="center")
    ax.text(0.44, 5.28, "input tokens and parameters are unchanged",
            fontsize=6.5, color=GRAY, ha="left", va="center")
    x0, token_w, token_gap = 0.72, 0.116, 0.012
    row_y = [4.63, 3.78, 2.93, 2.08]
    offsets = [0, 4, 8, 12]
    for row, (y, offset) in enumerate(zip(row_y, offsets)):
        ax.text(0.52, y + 0.12, rf"$r={offset}$", ha="right", va="center",
                fontsize=6.8, color=RED if row == 0 else BLUE,
                fontweight="bold" if row == 0 else "normal")
        for index in range(32):
            x = x0 + index * (token_w + token_gap)
            ax.add_patch(Rectangle(
                (x, y), token_w, 0.24,
                facecolor=PALE_RED if row == 0 else PALE_BLUE,
                edgecolor=WHITE, linewidth=0.25, zorder=1,
            ))
        for cut in range(offset, 33, 16):
            x = x0 + cut * (token_w + token_gap) - token_gap / 2
            ax.plot([x, x], [y - 0.07, y + 0.31],
                    color=RED if row == 0 else BLUE, linewidth=1.15, zorder=2)
        output_x = 5.14
        wire(ax, [(4.82, y + 0.12), (output_x - 0.12, y + 0.12)],
             color=RED if row == 0 else BLUE, lw=0.85)
        ax.add_patch(plt.Circle((output_x, y + 0.12), 0.13,
                                facecolor=PALE_RED if row == 0 else PALE_BLUE,
                                edgecolor=RED if row == 0 else BLUE,
                                linewidth=0.9, zorder=3))
        ax.text(output_x, y + 0.12, rf"$p_{{{offset}}}$",
                fontsize=5.8, color=DARK_RED if row == 0 else DARK_BLUE,
                ha="center", va="center", zorder=4)
    ax.text(2.72, 1.56,
            r"Each $r$ shifts the cuts, local RoPE phase, and token learning-rate phase",
            fontsize=6.25, color=GRAY, ha="center")

    ax.text(5.85, 5.72, "(b)  Confidence-cascaded marginalization",
            fontsize=8.2, fontweight="bold", color=INK, ha="left", va="center")
    box(ax, (5.92, 4.58), (1.52, 0.72), "Canonical pass", r"$p_0$  ·  frozen TTT",
        edge=GRAY, face=PALE_GRAY)
    box(ax, (7.88, 4.58), (1.28, 0.72), "Confidence", r"margin $\geq\tau$?",
        edge=GRAY, face=WHITE)
    box(ax, (10.30, 4.58), (1.34, 0.72), "Return", "one pass",
        edge=RED, face=PALE_RED, title_color=DARK_RED)
    wire(ax, [(7.44, 4.94), (7.88, 4.94)], color=GRAY)
    wire(ax, [(9.16, 4.94), (10.30, 4.94)], color=RED)
    ax.text(9.63, 5.05, "yes", fontsize=6.1, color=DARK_RED, ha="center")

    box(ax, (5.92, 3.12), (1.52, 0.72), "Opposite phase", r"evaluate $p_8$",
        edge=BLUE, face=PALE_BLUE, title_color=DARK_BLUE)
    box(ax, (7.88, 3.12), (1.28, 0.72), "Agreement", r"$\hat y_0=\hat y_8$?",
        edge=BLUE, face=WHITE, title_color=DARK_BLUE)
    box(ax, (9.58, 3.12), (2.06, 0.72), "Anchored fusion", "two passes",
        edge=RED, face=PALE_RED, title_color=DARK_RED)
    wire(ax, [(8.52, 4.58), (8.52, 4.18), (6.68, 4.18), (6.68, 3.84)], color=BLUE)
    ax.text(8.22, 4.29, "no", fontsize=6.1, color=DARK_BLUE, ha="center")
    wire(ax, [(7.44, 3.48), (7.88, 3.48)], color=BLUE)
    wire(ax, [(9.16, 3.48), (9.58, 3.48)], color=RED)
    ax.text(9.37, 3.59, "yes", fontsize=6.1, color=DARK_RED, ha="center")

    box(ax, (5.92, 1.66), (1.52, 0.72), "Quarter phases", r"evaluate $p_4,p_{12}$",
        edge=BLUE, face=PALE_BLUE, title_color=DARK_BLUE)
    box(ax, (7.88, 1.66), (3.76, 0.72), "Anchored four-phase fusion", "four passes",
        edge=RED, face=PALE_RED, title_color=DARK_RED)
    wire(ax, [(8.52, 3.12), (8.52, 2.72), (6.68, 2.72), (6.68, 2.38)], color=BLUE)
    ax.text(8.22, 2.83, "no", fontsize=6.1, color=DARK_BLUE, ha="center")
    wire(ax, [(7.44, 2.02), (7.88, 2.02)], color=RED)

    ax.text(8.78, 1.22,
            r"$\bar p_{\mathcal{S}}=|\mathcal{S}|^{-1}\sum_{s\in\mathcal{S}}p_s$"
            "     "
            r"$\widetilde p_0=(1-\rho)p_0+\rho\bar p_{\mathcal{S}}$",
            fontsize=7.1, color=INK, ha="center", va="center")
    ax.text(8.78, 0.78,
            r"locked values: $\tau=0.04$, $\rho=0.25$  ·  no weight, input, or state-rule change",
            fontsize=6.35, color=GRAY, ha="center", va="center")
    save(fig, "fig1_architecture")


def grouped_bars(ax, models, baseline, proposed, ylabel, ylim, reductions, pvalues):
    x = np.arange(len(models))
    width = 0.32
    bars0 = ax.bar(x - width / 2, baseline, width, color=GRAY, edgecolor=WHITE,
                   linewidth=0.45, label="Official phase", zorder=2)
    bars1 = ax.bar(x + width / 2, proposed, width, color=RED, edgecolor=WHITE,
                   linewidth=0.45, label="Confidence cascade", zorder=2)
    for bars in (bars0, bars1):
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value + ylim[1] * 0.025,
                    f"{value:.1f}", ha="center", va="bottom",
                    fontsize=6.3, color=INK)
    for index, (reduction, pvalue) in enumerate(zip(reductions, pvalues)):
        top = max(baseline[index], proposed[index])
        ax.text(index, top + ylim[1] * 0.125,
                f"−{reduction:.1f}%\n$p={pvalue}$", ha="center", va="bottom",
                fontsize=5.45, color=DARK_RED, fontweight="bold", linespacing=1.12)
    ax.set_xticks(x, models)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    style_axis(ax)


def figure_robustness(primary):
    models = ["350M", "760M"]
    raw_prediction = [100 * primary[m]["summary"]["raw"]["prediction_flip_rate"] for m in models]
    gate_prediction = [100 * primary[m]["summary"]["confidence_cascade"]["prediction_flip_rate"] for m in models]
    raw_correctness = [100 * primary[m]["summary"]["raw"]["correctness_flip_rate"] for m in models]
    gate_correctness = [100 * primary[m]["summary"]["confidence_cascade"]["correctness_flip_rate"] for m in models]
    pred_reduction = [100 * (a - b) / a for a, b in zip(raw_prediction, gate_prediction)]
    corr_reduction = [100 * (a - b) / a for a, b in zip(raw_correctness, gate_correctness)]
    pred_p = [r"1.5\times10^{-5}", r"7.6\times10^{-6}"]
    corr_p = [r"1.2\times10^{-4}", "0.0020"]

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.88),
                             gridspec_kw={"width_ratios": [1.0, 1.0, 0.94]})
    grouped_bars(axes[0], models, raw_prediction, gate_prediction,
                 "Prediction flips (%)", (0, 22), pred_reduction, pred_p)
    panel(axes[0], "(a)", "Prediction flips")
    grouped_bars(axes[1], models, raw_correctness, gate_correctness,
                 "Correctness flips (%)", (0, 12), corr_reduction, corr_p)
    panel(axes[1], "(b)", "Correctness flips")

    ax = axes[2]
    cascade_delta = [
        100 * (primary[m]["summary"]["confidence_cascade"]["default_accuracy"]
               - primary[m]["summary"]["raw"]["default_accuracy"])
        for m in models
    ]
    uniform_delta = [
        100 * (primary[m]["summary"]["uniform"]["accuracy"]
               - primary[m]["summary"]["raw"]["default_accuracy"])
        for m in models
    ]
    x = np.arange(2)
    width = 0.32
    for offset, values, color, label in (
        (-width / 2, cascade_delta, RED, "Cascade"),
        (width / 2, uniform_delta, BLUE, "Uniform 4-phase"),
    ):
        bars = ax.bar(x + offset, values, width, color=color, edgecolor=WHITE,
                      linewidth=0.45, label=label, zorder=2)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.045,
                    f"{value:+.2f}", ha="center", va="bottom",
                    fontsize=5.9, color=color, fontweight="bold")
    ax.axhline(0, color=MID_GRAY, linewidth=0.8, zorder=1)
    ax.set_xticks(x, models)
    ax.set_ylabel("Default accuracy change (pp)")
    ax.set_ylim(-0.25, 1.15)
    panel(ax, "(c)", "Default accuracy")
    style_axis(ax)
    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor=GRAY),
        Rectangle((0, 0), 1, 1, facecolor=RED),
        Rectangle((0, 0), 1, 1, facecolor=BLUE),
    ]
    fig.legend(legend_handles, ["Official phase", "Confidence cascade", "Uniform four-phase"],
               frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=3, handletextpad=0.45, columnspacing=1.0, fontsize=6.1)
    for ax in axes:
        ax.title.set_fontsize(7.5)
        ax.yaxis.label.set_size(6.9)
    fig.subplots_adjust(left=0.068, right=0.992, top=0.78, bottom=0.29, wspace=0.50)
    save(fig, "fig2_robustness")


def figure_efficiency(primary):
    models = ["350M", "760M"]
    histograms = [primary[m]["summary"]["timing"]["gated_phase_count_histogram"] for m in models]
    counts = np.array([[h[str(value)] / 384 * 100 for value in (1, 2, 4)] for h in histograms])
    means = [primary[m]["summary"]["timing"]["gated_mean_phase_count"] for m in models]

    fig = plt.figure(figsize=(TEXT_WIDTH, 3.12))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.02, 0.16, 0.92])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 2])]
    ax = axes[0]
    bottom = np.zeros(2)
    for index, (label, color) in enumerate((("1 pass", GRAY), ("2 passes", BLUE), ("4 passes", RED))):
        bars = ax.bar(np.arange(2), counts[:, index], bottom=bottom, width=0.56,
                      color=color, edgecolor=WHITE, linewidth=0.6, label=label, zorder=2)
        for model_index, bar in enumerate(bars):
            value = counts[model_index, index]
            if value >= 5:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bottom[model_index] + value / 2,
                        f"{value:.1f}%", ha="center", va="center",
                        fontsize=6.3, color=WHITE if index != 0 else INK,
                        fontweight="bold")
        bottom += counts[:, index]
    for index, value in enumerate(means):
        ax.text(index, 103.0, f"mean {value:.2f} passes", ha="center", va="bottom",
                fontsize=6.4, color=DARK_RED, fontweight="bold")
    ax.set_xticks(np.arange(2), models)
    ax.set_ylabel("Examples (%)")
    ax.set_ylim(0, 113)
    panel(ax, "(a)", "Most examples stop early")
    style_axis(ax)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=3, handletextpad=0.35, columnspacing=0.75)

    ax = axes[1]
    methods = ["Official phase", "Confidence cascade", "Anchored residual", "Uniform four-phase"]
    colors = [GRAY, RED, BLUE, DARK_BLUE]
    markers = ["o", "D", "s", "^"]

    def compact_latency(value):
        if value < 2.5:
            return value
        return 2.05 + 0.65 * (value - 4.90)

    for model_index, model in enumerate(models):
        s = primary[model]["summary"]
        t = s["timing"]
        x = [1.0, t["gated_latency_multiple"], t["full_latency_multiple"], t["full_latency_multiple"]]
        y = [100 * s["raw"]["prediction_flip_rate"],
             100 * s["confidence_cascade"]["prediction_flip_rate"],
             100 * s["anchored_residual"]["prediction_flip_rate"], 0.0]
        offset = -0.025 if model_index == 0 else 0.025
        for method_index, (xvalue, yvalue) in enumerate(zip(x, y)):
            ax.scatter(compact_latency(xvalue) + offset, yvalue,
                       s=27, marker=markers[method_index],
                       facecolor=colors[method_index] if model_index else WHITE,
                       edgecolor=colors[method_index], linewidth=1.0, zorder=3)
    ax.set_xlabel("Relative latency (×)")
    ax.set_ylabel("Prediction flips (%)")
    five_tick = compact_latency(5.0)
    ax.set_xlim(0.86, 2.27)
    ax.set_ylim(-0.55, 18.1)
    ax.set_xticks([1.0, 1.5, five_tick], ["1.0", "1.5", "5.0"])
    panel(ax, "(b)", "Robustness–cost trade-off")
    style_axis(ax, grid="both")
    for break_x in (1.82, 1.87):
        ax.plot([break_x - 0.018, break_x + 0.018], [-0.018, 0.018],
                transform=ax.get_xaxis_transform(), clip_on=False,
                color=MID_GRAY, linewidth=0.75)
    model_handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=5.2,
               markerfacecolor=WHITE, markeredgecolor=GRAY, label="350M (open)"),
        Line2D([], [], marker="o", linestyle="none", markersize=5.2,
               markerfacecolor=GRAY, markeredgecolor=GRAY, label="760M (filled)"),
    ]
    model_legend = ax.legend(handles=model_handles, frameon=False, loc="upper right",
                             fontsize=5.6, borderaxespad=0.25,
                             handletextpad=0.35, labelspacing=0.35)
    ax.add_artist(model_legend)
    method_handles = [
        Line2D([], [], marker=marker, linestyle="none", markersize=5.2,
               markerfacecolor=color, markeredgecolor=color, label=method)
        for method, color, marker in zip(methods, colors, markers)
    ]
    ax.legend(handles=method_handles, frameon=False, loc="upper center",
              bbox_to_anchor=(0.50, -0.235), ncol=2, fontsize=5.65,
              handletextpad=0.28, columnspacing=0.62, labelspacing=0.40)
    memory_350 = 100 * (primary["350M"]["summary"]["timing"]["gated_peak_allocated_gib"] /
                        primary["350M"]["summary"]["timing"]["baseline_peak_allocated_gib"] - 1)
    memory_760 = 100 * (primary["760M"]["summary"]["timing"]["gated_peak_allocated_gib"] /
                        primary["760M"]["summary"]["timing"]["baseline_peak_allocated_gib"] - 1)
    ax.text(1.08, 4.65,
            f"Peak memory overhead\n+{memory_350:.2f}% (350M); +{memory_760:.2f}% (760M)",
            ha="left", va="center", fontsize=5.05, color=GRAY, linespacing=1.20)
    for ax in axes:
        ax.title.set_fontsize(7.45)
        ax.yaxis.label.set_size(7.0)
        ax.xaxis.label.set_size(7.0)
    fig.subplots_adjust(left=0.073, right=0.992, top=0.80, bottom=0.37, wspace=0.0)
    save(fig, "fig3_efficiency")


def figure_external():
    evidence = read_json("polyphase_disjoint_final_analysis.json")
    runtime_files = {
        "SQuAD": read_json("polyphase_fast_benchmark_squad_350m_n24.json"),
        "HotpotQA": read_json("polyphase_fast_benchmark_hotpot_760m_n24.json"),
    }
    labels = ["SQuAD\nheld-out (n=61)", "HotpotQA\nexternal (n=62)"]
    cells = [evidence["heldout_squad"]["metrics"], evidence["external_hotpotqa"]["metrics"]]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.30),
                             gridspec_kw={"width_ratios": [1.18, 0.92]})
    ax = axes[0]
    x = np.arange(2)
    width = 0.18
    raw_prediction = [100 * cell["raw"]["prediction_flip_rate"] for cell in cells]
    gate_prediction = [100 * cell["prediction_flip_rate"] for cell in cells]
    raw_correctness = [100 * cell["raw"]["correctness_flip_rate"] for cell in cells]
    gate_correctness = [100 * cell["correctness_flip_rate"] for cell in cells]
    series = [
        (-1.5 * width, raw_prediction, GRAY, "Prediction · official"),
        (-0.5 * width, gate_prediction, RED, "Prediction · cascade"),
        (0.5 * width, raw_correctness, MID_GRAY, "Correctness · official"),
        (1.5 * width, gate_correctness, BLUE, "Correctness · cascade"),
    ]
    for offset, values, color, label in series:
        bars = ax.bar(x + offset, values, width, color=color, edgecolor=WHITE,
                      linewidth=0.4, label=label, zorder=2)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.22,
                    f"{value:.1f}", ha="center", va="bottom",
                    fontsize=5.7, color=INK)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Cross-origin flip rate (%)")
    ax.set_ylim(0, 10.2)
    panel(ax, "(a)", "Independent QA checks")
    style_axis(ax)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25),
              ncol=2, handletextpad=0.35, columnspacing=0.8)

    ax = axes[1]
    models = ["SQuAD\n350M", "HotpotQA\n760M"]
    gate_latency = [runtime_files[name]["results"]["gated"]["latency_multiple"] for name in ("SQuAD", "HotpotQA")]
    full_latency = [runtime_files[name]["results"]["full"]["latency_multiple"] for name in ("SQuAD", "HotpotQA")]
    width = 0.31
    for offset, values, color, label in (
        (-width / 2, gate_latency, RED, "Cascade"),
        (width / 2, full_latency, BLUE, "Full four-phase"),
    ):
        bars = ax.bar(np.arange(2) + offset, values, width, color=color,
                      edgecolor=WHITE, linewidth=0.45, label=label, zorder=2)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.12,
                    f"{value:.2f}×", ha="center", va="bottom",
                    fontsize=6.1, color=color, fontweight="bold")
    ax.axhline(1, color=MID_GRAY, linewidth=0.8, zorder=1)
    ax.set_xticks(np.arange(2), models)
    ax.set_ylabel("Measured latency multiple")
    ax.set_ylim(0, 5.8)
    panel(ax, "(b)", "Small-screen runtime")
    style_axis(ax)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25),
              ncol=2, handletextpad=0.4)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.79, bottom=0.34, wspace=0.31)
    save(fig, "figS1_external")


def main():
    primary, _ = prepare_data()
    figure_architecture()
    figure_robustness(primary)
    figure_efficiency(primary)
    print(FIGURES)


if __name__ == "__main__":
    main()
