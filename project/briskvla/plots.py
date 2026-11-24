"""Report figures for BriskVLA, in the figures4papers house style (github.com/ChenLiu-1996/figures4papers).

Reads the raw result files under outputs/ and writes PDF + PNG figures to project/briskvla/figures/:

    uv run --with matplotlib python project/briskvla/plots.py            # all figures
    uv run --with matplotlib python project/briskvla/plots.py distill    # only names containing "distill"

Data sources (all gitignored, on the laptop):
    outputs/latency/fuse_vs_drop.json          latency breakdown, teacher / k8_drop / k8_fuse at K = 1, 3, 10
    outputs/latency/breakdown_teacher.json     FFN width microbenchmark (ffn_width_sweep_us)
    outputs/zeroshot/<name>/metrics.jsonl      step-0 screening on the 60-episode subset
    outputs/distill/<run>/metrics.jsonl        copies of the Colab distill runs (Drive: BriskVLA/distill/<run>)

Latency numbers come from an RTX 5080 Laptop GPU; the figures say so. Swap in the RTX 3080 files once measured.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
FIG = Path(__file__).resolve().parent / "figures"

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_line": "#D88F8A",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "gray": "#767676",
    "dark": "#4D4D4D",
    "highlight": "#FFD700",
    "teal": "#42949E",
    "violet": "#9A4D8E",
}

# One colour per model, used in every figure. Blue = ours, green / red = baselines, gray = teacher.
MODELS = {
    "teacher": ("Teacher (16L)", PALETTE["dark"]),
    "pairs": ("Student — 8L Spread", PALETTE["green_3"]),
    "k8_drop": ("Student — 8L Alternating", PALETTE["red_line"]),
    "k8_fuse": ("Student — 8L Alternating + FFN Fusion", PALETTE["blue_main"]),
}
GPU_NOTE = "RTX 5080 Laptop, batch 1"
DOWN = "  ←"  # y labels are rotated 90° CCW, so this reads as ↓ (lower is better)
TEACHER_FLOOR = 0.0266  # teacher's own chunk_mse_gt on the 85-episode held-out set


def apply_style(font_size: int = 16, linewidth: float = 2.0) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "Inter", "DejaVu Sans"],
            "font.size": font_size,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": linewidth,
            "legend.frameon": False,
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save(fig: plt.Figure, name: str, pad: float = 2.0) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=pad)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"wrote {FIG / name}.{{pdf,png}}")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def latency() -> dict:
    return json.loads((OUT / "latency" / "fuse_vs_drop.json").read_text())["models"]


# --------------------------------------------------------------------------------------------------
# Concept figures (no data)
# --------------------------------------------------------------------------------------------------


def _box(ax, x, y, w, h, text, color, fontsize=14, edge="black"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=color,
            edgecolor=edge,
            linewidth=1.8,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def _arrow(ax, p, q, lw=1.8, color="black", style="-|>", ms=14):
    ax.add_patch(
        FancyArrowPatch(
            p, q, arrowstyle=style, mutation_scale=ms, linewidth=lw, color=color, shrinkA=0, shrinkB=0
        )
    )


R_ADD = 0.13  # radius of the residual-add circle


def _add(ax, x, y):
    ax.add_patch(Circle((x, y), R_ADD, facecolor="white", edgecolor="black", linewidth=1.6, zorder=3))
    ax.plot([x - R_ADD * 0.6, x + R_ADD * 0.6], [y, y], color="black", lw=1.6, zorder=4)
    ax.plot([x, x], [y - R_ADD * 0.6, y + R_ADD * 0.6], color="black", lw=1.6, zorder=4)


def _skip(ax, x, y_from, y_add, x_out):
    """Residual path: branch off the main column at y_from, go right to x_out, up, and into the add at y_add."""
    ax.plot([x, x_out, x_out], [y_from, y_from, y_add], color=PALETTE["gray"], lw=1.6)
    _arrow(ax, (x_out, y_add), (x + R_ADD, y_add), lw=1.6, color=PALETTE["gray"])


def plot_method() -> None:
    """Teacher layer pair (a, b) -> one fused layer with one attention block and FFN_a || FFN_b."""
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13.4)
    ax.set_ylim(-1.15, 6.35)
    ax.set_axis_off()
    attn_c, ffa_c, ffb_c = PALETTE["blue_secondary"] + "55", PALETTE["green_2"], PALETTE["red_1"]
    h, pitch, add_dy = 0.6, 1.25, 0.9  # box height, distance between blocks, box bottom -> add centre

    # Teacher: x -> [Attn_a, FFN_a, Attn_b, FFN_b], each a residual block.
    cx, w = 2.4, 2.9
    labels = [
        ("Attention$_a$ (joint)", attn_c),
        ("FFN$_a$   720→2048→720", ffa_c),
        ("Attention$_b$ (cross)", attn_c),
        ("FFN$_b$   720→2048→720", ffb_c),
    ]
    y = 0.3
    _arrow(ax, (cx, -0.3), (cx, y))
    for t, c in labels:
        _box(ax, cx - w / 2, y, w, h, t, c)
        _arrow(ax, (cx, y + h), (cx, y + add_dy - R_ADD))
        _add(ax, cx, y + add_dy)
        _skip(ax, cx, y - 0.12, y + add_dy, cx + w / 2 + 0.3)
        nxt = y + pitch
        _arrow(ax, (cx, y + add_dy + R_ADD), (cx, nxt if t != labels[-1][0] else y + add_dy + 0.6))
        y = nxt
    ax.text(cx - 0.12, -0.3, "$x$", ha="right", va="bottom", fontsize=16)
    ax.text(cx, 5.95, "Teacher: two layers (depths $2i$, $2i{+}1$)", ha="center", fontsize=16, weight="bold")

    # Merge arrow
    _arrow(ax, (4.75, 2.6), (6.25, 2.6), lw=3, ms=26, color=PALETTE["blue_main"])
    ax.text(5.5, 2.85, "merge", ha="center", fontsize=15, color=PALETTE["blue_main"], weight="bold")
    ax.text(
        5.5,
        2.35,
        "fold RMSNorm gains\ninto FFN weights,\nstack hidden units",
        ha="center",
        va="top",
        fontsize=11.5,
        color=PALETTE["gray"],
    )

    # Fused: x -> Attn_keep (residual) -> FFN_a || FFN_b, both added to the residual.
    fx = 9.3
    ya = 0.3 + pitch * 0.5
    _arrow(ax, (fx, -0.3), (fx, ya))
    ax.text(fx - 0.12, -0.3, "$x$", ha="right", va="bottom", fontsize=16)
    _box(ax, fx - 1.45, ya, 2.9, h, "Attention$_{keep}$", attn_c)
    _arrow(ax, (fx, ya + h), (fx, ya + add_dy - R_ADD))
    _add(ax, fx, ya + add_dy)
    _skip(ax, fx, ya - 0.12, ya + add_dy, fx + 1.75)
    yf = ya + pitch + 0.25
    split = yf - 0.22
    ax.plot([fx, fx], [ya + add_dy + R_ADD, split], color="black", lw=1.8)
    ax.plot([fx - 1.0, fx + 1.0], [split, split], color="black", lw=1.8)
    for dx, t, c in [(-1.0, "FFN$_a$", ffa_c), (1.0, "FFN$_b$", ffb_c)]:
        _arrow(ax, (fx + dx, split), (fx + dx, yf))
        _box(ax, fx + dx - 0.75, yf, 1.5, h, t, c)
    y_add = yf + h + 0.55
    for dx in (-1.0, 1.0):
        _arrow(ax, (fx + dx, yf + h), (fx + np.sign(dx) * R_ADD * 0.7, y_add - R_ADD * 0.7))
    _add(ax, fx, y_add)
    _skip(ax, fx, split - 0.1, y_add, fx + 2.15)
    ax.add_patch(
        Rectangle(
            (fx - 1.9, yf - 0.1), 3.8, h + 0.2, fill=False, ls="--", lw=1.4, edgecolor=PALETTE["blue_main"]
        )
    )
    ax.text(
        fx + 2.3,
        yf + h / 2,
        "one SwiGLU\n720→4096→720",
        ha="left",
        va="center",
        fontsize=12,
        color=PALETTE["blue_main"],
    )
    _arrow(ax, (fx, y_add + R_ADD), (fx, y_add + 0.75))
    ax.text(fx, 5.95, "Student: one fused layer", ha="center", fontsize=16, weight="bold")

    ax.text(
        6.7,
        -0.85,
        r"$x + f_a(x) + f_b(x + f_a(x))\ \approx\ x + f_a(x) + f_b(x)$"
        "   (holds when $\\|f_a(x)\\| \\ll \\|x\\|$; the weight merge itself is exact)",
        ha="center",
        fontsize=13,
    )
    save(fig, "fig_method_fusion", pad=0.5)


def plot_layouts() -> None:
    """Which of the 16 VLM depths keep an expert layer, per model; joint vs cross attention by depth."""
    rows = [
        ("Teacher (16L)", list(range(16)), False),
        ("8L Spread", [0, 1, 4, 5, 8, 9, 12, 13], False),
        ("8L Alternating", [0, 3, 4, 7, 8, 11, 12, 15], False),
        ("8L Alternating + FFN Fusion", [0, 3, 4, 7, 8, 11, 12, 15], True),
    ]
    joint_c, cross_c = PALETTE["blue_secondary"], PALETTE["teal"] + "AA"
    fig, ax = plt.subplots(figsize=(14, 5.2))
    gap = 0.35  # extra space between teacher pairs
    xs = [d + gap * (d // 2) for d in range(16)]
    # FFNs are coloured by their position in the teacher pair (2i, 2i+1), so a reader can track each one.
    ffn_c = {0: PALETTE["green_2"], 1: PALETTE["red_1"]}
    cw, fy, fh = 0.9, 0.1, 0.34  # cell width; FFN block y offset and height (attention sits above it)
    for r, (name, kept, fused) in enumerate(rows):
        y = len(rows) - 1 - r
        for d in range(16):
            x = xs[d]
            if d not in kept:
                ax.add_patch(
                    Rectangle(
                        (x, y + 0.1), cw, 0.8, facecolor="white", edgecolor=PALETTE["neutral"], linewidth=1.0
                    )
                )
                if fused:  # this layer's FFN was moved into its partner: dashed ghost + arrow
                    ax.add_patch(
                        Rectangle(
                            (x + 0.08, y + fy + 0.04),
                            cw - 0.16,
                            fh - 0.08,
                            facecolor="none",
                            edgecolor=PALETTE["red_strong"] if d % 2 else "#4E9A4E",
                            ls="--",
                            linewidth=1.4,
                        )
                    )
                    partner = d + 1 if d % 2 == 0 else d - 1
                    half = (cw - 0.04) / 2
                    tx = xs[partner] + (half / 2 if d % 2 == 0 else cw - half / 2)
                    ax.add_patch(
                        FancyArrowPatch(
                            (x + cw / 2, y + fy + 0.02),
                            (tx, y + fy - 0.01),
                            connectionstyle=f"arc3,rad={-0.6 if d > partner else 0.6}",
                            arrowstyle="-|>",
                            mutation_scale=11,
                            linewidth=1.4,
                            color=PALETTE["dark"],
                            shrinkA=0,
                            shrinkB=0,
                        )
                    )
                continue
            ax.add_patch(
                Rectangle(
                    (x, y + 0.48),
                    cw,
                    0.42,
                    facecolor=joint_c if d % 2 == 0 else cross_c,
                    edgecolor="black",
                    linewidth=1.5,
                )
            )
            if fused:  # one attention block, both FFNs of the pair side by side (run as one wide SwiGLU)
                half = (cw - 0.04) / 2
                ax.add_patch(
                    Rectangle((x, y + fy), half, fh, facecolor=ffn_c[0], edgecolor="black", linewidth=1.5)
                )
                ax.add_patch(
                    Rectangle(
                        (x + half + 0.04, y + fy),
                        half,
                        fh,
                        facecolor=ffn_c[1],
                        edgecolor="black",
                        linewidth=1.5,
                    )
                )
            else:
                ax.add_patch(
                    Rectangle((x, y + fy), cw, fh, facecolor=ffn_c[d % 2], edgecolor="black", linewidth=1.5)
                )
        ax.text(
            -0.4,
            y + 0.5,
            name,
            ha="right",
            va="center",
            fontsize=15,
            color=PALETTE["blue_main"] if fused else "black",
            weight="bold" if fused else "normal",
        )
    for d in range(16):
        ax.text(xs[d] + 0.45, -0.25, str(d), ha="center", va="center", fontsize=12, color=PALETTE["gray"])
    ax.text(xs[-1] / 2 + 0.45, -0.75, "VLM depth (all 16 VLM layers always run)", ha="center", fontsize=14)
    ax.set_xlim(-7.5, xs[-1] + 1.1)
    ax.set_ylim(-1.0, len(rows) + 0.3)
    ax.set_axis_off()
    handles = [
        Patch(facecolor=joint_c, edgecolor="black", label="joint attention (even depth)"),
        Patch(facecolor=cross_c, edgecolor="black", label="cross-attention (odd depth)"),
        Patch(facecolor="white", edgecolor=PALETTE["neutral"], label="no expert layer"),
        Patch(facecolor=ffn_c[0], edgecolor="black", label="FFN of 1st layer in pair (depth 2i)"),
        Patch(facecolor=ffn_c[1], edgecolor="black", label="FFN of 2nd layer in pair (depth 2i+1)"),
        Patch(facecolor="none", edgecolor=PALETTE["dark"], ls="--", label="FFN moved into the fused layer"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        ncols=3,
        fontsize=12,
        handlelength=1.2,
        columnspacing=1.5,
    )
    save(fig, "fig_layouts", pad=0.5)


def plot_distill_diagram() -> None:
    """Teacher -> shallow student distillation (the `pairs` student): which layers exist, which VLM layers each
    expert layer reads, and where each loss attaches. Student is mirrored so the expert columns face each other."""
    kept = [0, 1, 4, 5, 8, 9, 12, 13]  # student expert layers (VLM depths), haozh/distill `pairs` layout
    match = {
        d: 2 * j + 1 for j, d in enumerate(kept)
    }  # --hidden_map uniform: student layer j -> teacher depth 2j+1
    fig, ax = plt.subplots(figsize=(14, 11.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(-1.15, 12.3)
    ax.set_axis_off()

    vlm_c, joint_c, cross_c = (
        "#EEEEEE",
        "#BCD1E8",
        "#C0DBDF",
    )  # light blue / teal, opaque so lines stay behind
    hid_c, h = PALETTE["violet"], 0.42
    tv, te = (0.9, 2.2), (3.2, 1.8)  # teacher VLM / expert columns: (x0, width)
    se, sv = (9.0, 1.8), (10.9, 2.2)  # student expert / VLM columns

    def yb(d):  # bottom of the row at VLM depth d
        return 1.6 + d * 0.52

    def block(x0, w, d, text, face, edge="black", lw=1.3, color="black"):
        ax.add_patch(
            FancyBboxPatch(
                (x0, yb(d)),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=0.06",
                facecolor=face,
                edgecolor=edge,
                linewidth=lw,
            )
        )
        ax.text(x0 + w / 2, yb(d) + h / 2, text, ha="center", va="center", fontsize=10.5, color=color)

    # Student residual stream: depths without an expert layer pass the action tokens through unchanged.
    sx = se[0] + se[1] / 2
    ax.plot([sx, sx], [yb(0) - 0.25, yb(15) + h], color=PALETTE["gray"], lw=1.6, zorder=0)

    for d in range(16):
        yc = yb(d) + h / 2
        ex_c = joint_c if d % 2 == 0 else cross_c
        # teacher
        block(*tv, d, f"VLM layer {d}", vlm_c, edge=PALETTE["gray"])
        matched = d in match.values()
        block(*te, d, f"expert {d}", ex_c, edge=hid_c if matched else "black", lw=2.4 if matched else 1.3)
        ax.text(tv[0] - 0.15, yc, str(d), ha="right", va="center", fontsize=11, color=PALETTE["gray"])
        # student
        if d in kept:
            block(*sv, d, f"VLM layer {d}", vlm_c, edge=PALETTE["gray"])
            block(*se, d, f"student {kept.index(d)}", ex_c, edge=hid_c, lw=2.4)
        else:
            block(*sv, d, f"VLM layer {d}", "white", edge=PALETTE["neutral"], color=PALETTE["neutral"])
        ax.text(sv[0] + sv[1] + 0.15, yc, str(d), ha="left", va="center", fontsize=11, color=PALETTE["gray"])
    ax.text(
        sx,
        yb(2) + 0.5,
        "identity\n(no expert layer)",
        ha="center",
        va="center",
        fontsize=10,
        color=PALETTE["gray"],
        style="italic",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
    )
    ax.text(
        tv[0] - 0.55,
        yb(8),
        "VLM depth",
        rotation=90,
        ha="center",
        va="center",
        fontsize=12,
        color=PALETTE["gray"],
    )

    # Hidden-state loss: student layer j <-> teacher expert depth 2j+1.
    for d, t in match.items():
        p, q = (se[0], yb(d) + h / 2), (te[0] + te[1], yb(t) + h / 2)
        ax.plot([p[0], q[0]], [p[1], q[1]], color=hid_c, lw=1.5, alpha=0.85)
        ax.plot(*zip(p, q, strict=True), "o", color=hid_c, ms=4.5)
    ax.text(
        7.05,
        yb(6.6),
        "$\\mathcal{L}_{hidden}$",
        ha="center",
        va="center",
        fontsize=19,
        color=hid_c,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 2},
    )
    ax.text(
        7.05,
        yb(5.65),
        "student layer $j$ ↔ teacher\nexpert depth $2j{+}1$\n(layer-normed, action tokens)",
        ha="center",
        va="center",
        fontsize=10.5,
        color=hid_c,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 2},
    )

    # Inputs (bottom).
    yi = yb(0) - 0.6
    for x0, w, n, face, dots in [
        (tv[0], tv[1], 5, vlm_c, True),
        (sv[0], sv[1], 5, vlm_c, True),
        (te[0], te[1], 4, cross_c, False),
        (se[0], se[1], 4, cross_c, False),
    ]:
        xs_ = np.linspace(x0 + 0.25, x0 + w - 0.25, n)
        for i, x in enumerate(xs_):
            if dots and i == n - 2:
                ax.text(x, yi, "…", ha="center", va="center", fontsize=13)
            else:
                ax.add_patch(Circle((x, yi), 0.13, facecolor=face, edgecolor="black", linewidth=1.2))
    for x0, w, lab in [
        (tv[0], tv[1], "2 images, instruction, state"),
        (te[0], te[1], "noisy actions $x_t$, time $t$"),
        (se[0], se[1], "noisy actions $x_t$, time $t$"),
        (sv[0], sv[1], "2 images, instruction, state"),
    ]:
        ax.text(x0 + w / 2, yi - 0.3, lab, ha="center", va="top", fontsize=10.5)
    _arrow(ax, (te[0] + te[1] + 0.15, yi), (se[0] - 0.15, yi), color=PALETTE["gray"], style="<|-|>", lw=1.4)
    ax.text(
        7.05, yi + 0.12, "same noise and $t$", ha="center", va="bottom", fontsize=10.5, color=PALETTE["gray"]
    )

    # Outputs (top): velocity at the same (x_t, t), and the full chunk after 10 Euler steps.
    yo = yb(15) + h + 0.55
    for x0, w, lab, side in [(te[0], te[1], "$v_\\phi$", "left"), (se[0], se[1], "$v_\\theta$", "right")]:
        cx_ = x0 + w / 2
        _arrow(ax, (cx_, yb(15) + h), (cx_, yo - 0.16))
        for x in np.linspace(x0 + 0.3, x0 + w - 0.3, 4):
            ax.add_patch(Circle((x, yo), 0.13, facecolor=cross_c, edgecolor="black", linewidth=1.2))
        ax.text(
            x0 - 0.12 if side == "left" else x0 + w + 0.12,
            yo,
            lab,
            ha="right" if side == "left" else "left",
            va="center",
            fontsize=19,
        )
    _arrow(ax, (te[0] + te[1] + 0.15, yo), (se[0] - 0.15, yo), style="<|-|>", lw=1.6)
    ax.text(7.05, yo + 0.1, "$\\mathcal{L}_{vel}$", ha="center", va="bottom", fontsize=19)
    yc_ = yo + 0.95
    ax.text(
        te[0] + te[1] / 2,
        yc_,
        "action chunk $\\hat{A}_\\phi$\n10 Euler steps, no grad",
        ha="center",
        va="center",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "black"},
    )
    ax.text(
        se[0] + se[1] / 2,
        yc_,
        "action chunk $\\hat{A}_\\theta$\n10 Euler steps, backprop",
        ha="center",
        va="center",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "black"},
    )
    _arrow(ax, (te[0] + te[1] + 1.0, yc_), (se[0] - 1.0, yc_), style="<|-|>", lw=1.6)
    ax.text(7.05, yc_ + 0.1, "$\\mathcal{L}_{out}$", ha="center", va="bottom", fontsize=19)

    # Titles and loss.
    ax.text(
        (tv[0] + te[0] + te[1]) / 2,
        12.15,
        "Teacher: SmolVLA, 16 expert layers (frozen)",
        ha="center",
        va="center",
        fontsize=15,
        weight="bold",
    )
    ax.text(
        (se[0] + sv[0] + sv[1]) / 2,
        12.15,
        "Student: 8 expert layers (trained)",
        ha="center",
        va="center",
        fontsize=15,
        weight="bold",
        color=PALETTE["blue_main"],
    )
    ax.text(
        7.0,
        -0.6,
        "$\\mathcal{L} = \\mathcal{L}_{vel} + 0.1\\,\\mathcal{L}_{hidden} + \\mathcal{L}_{out}$"
        "   (flow-matching loss on dataset actions: weight 0)",
        ha="center",
        va="center",
        fontsize=14,
    )
    ax.text(
        7.0,
        -0.95,
        "VLM is frozen and identical in both: its prefix KV cache is computed once per batch and shared."
        "  Expert layer at depth $d$ reads VLM layer $d$.",
        ha="center",
        va="center",
        fontsize=11,
        color=PALETTE["gray"],
    )
    handles = [
        Patch(facecolor=joint_c, edgecolor="black", label="expert: joint attention (even depth)"),
        Patch(facecolor=cross_c, edgecolor="black", label="expert: cross-attention (odd depth)"),
        Patch(facecolor=vlm_c, edgecolor=PALETTE["gray"], label="VLM layer read by an expert layer"),
        Patch(facecolor="white", edgecolor=PALETTE["neutral"], label="VLM layer not read (still runs)"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(7.0, 0.3),
        bbox_transform=ax.transData,
        ncols=4,
        fontsize=10.5,
        handlelength=1.2,
        columnspacing=1.2,
    )
    save(fig, "fig_distill_diagram", pad=0.4)


def plot_distill_diagram_small() -> None:
    """Compact teacher -> `pairs` student diagram (about 7 x 3.3 in, for a full-width figure in a tight paper)."""
    kept = [0, 1, 4, 5, 8, 9, 12, 13]
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    ax.set_xlim(0, 14)
    ax.set_ylim(-0.35, 6.45)
    ax.set_axis_off()
    vlm_c, act_c, hid_c, fs = "#E4ECF7", "#C9EDE6", PALETTE["violet"], 7.5

    def bar(x0, w, y, hh, face, edge="#555555", lw=0.7):
        ax.add_patch(
            FancyBboxPatch(
                (x0, y),
                w,
                hh,
                boxstyle="round,pad=0,rounding_size=0.05",
                facecolor=face,
                edgecolor=edge,
                linewidth=lw,
            )
        )

    def arrow(p, q, style="-|>", color="black", lw=0.8):
        ax.add_patch(
            FancyArrowPatch(
                p, q, arrowstyle=style, mutation_scale=7, linewidth=lw, color=color, shrinkA=0, shrinkB=0
            )
        )

    # Teacher: 16 rows of [VLM layer d | expert layer d].
    tv, te = (0.9, 2.1), (3.05, 1.45)
    ty0, tp, th = 0.95, 0.3, 0.24

    def ty(d):
        return ty0 + d * tp

    matched = {2 * j + 1 for j in range(len(kept))}
    for d in range(16):
        bar(*tv, ty(d), th, vlm_c)
        bar(*te, ty(d), th, act_c, edge=hid_c if d in matched else "#555555", lw=1.2 if d in matched else 0.7)
        if d % 5 == 0 or d == 15:
            ax.text(
                tv[0] - 0.08,
                ty(d) + th / 2,
                str(d),
                ha="right",
                va="center",
                fontsize=6,
                color=PALETTE["gray"],
            )
    # bracket "L = 16"
    bx = tv[0] - 0.42
    ax.plot([bx + 0.12, bx, bx, bx + 0.12], [ty(0), ty(0), ty(15) + th, ty(15) + th], color="black", lw=0.8)
    ax.text(bx - 0.08, (ty(0) + ty(15) + th) / 2, "L=16", ha="right", va="center", fontsize=fs)

    # Student (mirrored): 8 rows of [student layer j | VLM layer it reads].
    se, sv = (8.75, 1.45), (10.25, 1.75)
    sy0, sp, sh = 1.75, 0.42, 0.34

    def sy(j):
        return sy0 + j * sp

    for j, d in enumerate(kept):
        bar(*se, sy(j), sh, act_c, edge=hid_c, lw=1.2)
        bar(*sv, sy(j), sh, vlm_c)
        ax.text(se[0] + se[1] / 2, sy(j) + sh / 2, f"layer {j}", ha="center", va="center", fontsize=6.5)
        ax.text(sv[0] + sv[1] / 2, sy(j) + sh / 2, f"VLM {d}", ha="center", va="center", fontsize=6.5)
        # hidden-state loss: student layer j <-> teacher expert depth 2j+1
        ax.plot(
            [se[0], te[0] + te[1]], [sy(j) + sh / 2, ty(2 * j + 1) + th / 2], color=hid_c, lw=0.8, alpha=0.9
        )
    bx = sv[0] + sv[1] + 0.3
    ax.plot([bx - 0.12, bx, bx, bx - 0.12], [sy(0), sy(0), sy(7) + sh, sy(7) + sh], color="black", lw=0.8)
    ax.text(bx + 0.08, (sy(0) + sy(7) + sh) / 2, "L=8", ha="left", va="center", fontsize=fs)
    # One white backing box for both label lines, so neither text clips the other.
    ax.add_patch(
        FancyBboxPatch(
            (5.25, ty(5.6)),
            2.6,
            ty(8.05) - ty(5.6),
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor="white",
            edgecolor="none",
            zorder=3,
        )
    )
    ax.text(
        6.55,
        ty(7.45),
        "$\\mathcal{L}_{hidden}$",
        ha="center",
        va="center",
        fontsize=10,
        color=hid_c,
        zorder=4,
    )
    ax.text(
        6.55,
        ty(6.1),
        "layer $j$ ↔ depth $2j{+}1$",
        ha="center",
        va="center",
        fontsize=6.5,
        color=hid_c,
        zorder=4,
    )

    # Outputs: velocity (L_vel) and 10-step action chunk (L_out).
    yo = ty(16) + 0.42
    tcx, scx = te[0] + te[1] / 2, se[0] + se[1] / 2
    for cx_, y_from, lab, side in [(tcx, ty(15) + th, "$v_\\phi$", -1), (scx, sy(7) + sh, "$v_\\theta$", 1)]:
        arrow((cx_, y_from), (cx_, yo - 0.13))
        for x in np.linspace(cx_ - 0.5, cx_ + 0.5, 4):
            ax.add_patch(Circle((x, yo), 0.1, facecolor=act_c, edgecolor="#555555", linewidth=0.7))
        ax.text(cx_ + side * 0.68, yo, lab, ha="left" if side > 0 else "right", va="center", fontsize=11)
    arrow((tcx + 1.15, yo), (scx - 1.15, yo), style="<|-|>")
    ax.text(
        6.55, yo + 0.06, "$\\mathcal{L}_{vel}$,  $\\mathcal{L}_{out}$", ha="center", va="bottom", fontsize=10
    )

    # Inputs and titles.
    yi = 0.62
    for x0, w, n, face in [(tv[0], tv[1], 5, vlm_c), (te[0], te[1], 4, act_c)]:
        for i, x in enumerate(np.linspace(x0 + 0.2, x0 + w - 0.2, n)):
            if face is vlm_c and i == n - 2:
                ax.text(x, yi, "…", ha="center", va="center", fontsize=7)
            else:
                ax.add_patch(Circle((x, yi), 0.1, facecolor=face, edgecolor="#555555", linewidth=0.7))
    sy_in = sy(0) - 0.32
    for x0, w, n, face in [(se[0], se[1], 4, act_c), (sv[0], sv[1], 5, vlm_c)]:
        for i, x in enumerate(np.linspace(x0 + 0.2, x0 + w - 0.2, n)):
            if face is vlm_c and i == n - 2:
                ax.text(x, sy_in, "…", ha="center", va="center", fontsize=7)
            else:
                ax.add_patch(Circle((x, sy_in), 0.1, facecolor=face, edgecolor="#555555", linewidth=0.7))
    ax.text(tv[0] + tv[1] / 2, yi - 0.2, "V-L tokens", ha="center", va="top", fontsize=6.5)
    ax.text(te[0] + te[1] / 2, yi - 0.2, "action tokens", ha="center", va="top", fontsize=6.5)
    ax.text(se[0] + se[1] / 2, sy_in - 0.2, "action tokens", ha="center", va="top", fontsize=6.5)
    ax.text(sv[0] + sv[1] / 2, sy_in - 0.2, "V-L tokens", ha="center", va="top", fontsize=6.5)
    ax.text((tv[0] + te[0] + te[1]) / 2, -0.22, "Teacher (frozen)", ha="center", va="center", fontsize=10)
    ax.text(
        (se[0] + sv[0] + sv[1]) / 2,
        -0.22,
        "Student (trained)",
        ha="center",
        va="center",
        fontsize=10,
        color=PALETTE["blue_main"],
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(
            FIG / f"fig_distill_diagram_small.{ext}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
            facecolor="white",
        )
    plt.close(fig)
    print(f"wrote {FIG / 'fig_distill_diagram_small'}.{{pdf,png}}")


def plot_distill_diagram_paper() -> None:
    """Single-column (3.25 in) teacher -> spread student diagram for the paper's placement runs (haozh/distill
    defaults): L = L_vel + 0.1 L_hid + L_out, ground-truth term off, --hidden_map same_depth (each student layer is
    matched to the teacher layer it was copied from). Student rows sit at their teacher depth, so L_hid lines are
    horizontal. Axes are in inches so the PDF is exactly 3.25 in wide."""
    kept = [0, 1, 4, 5, 8, 9, 12, 13]
    width_in, height_in = 3.25, 2.95
    fig = plt.figure(figsize=(width_in, height_in))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width_in)
    ax.set_ylim(0, height_in)
    ax.set_axis_off()
    vlm_c, act_c, hid_c, edge = "#E4ECF7", "#C9EDE6", PALETTE["violet"], "#555555"
    tv, te = (0.3, 0.5), (0.85, 0.44)  # teacher VLM / expert columns (x0, width), inches
    se, sv = (2.0, 0.44), (2.49, 0.5)  # student expert / VLM columns (mirrored)
    y0, pitch, rh = 0.7, 0.12, 0.094
    gx = (te[0] + te[1] + se[0]) / 2  # centre of the gap between teacher and student

    def yb(d):
        return y0 + d * pitch

    def bar(x0, w, d, face, ec=edge, lw=0.5, ls="-"):
        ax.add_patch(
            FancyBboxPatch(
                (x0, yb(d)),
                w,
                rh,
                boxstyle="round,pad=0,rounding_size=0.015",
                facecolor=face,
                edgecolor=ec,
                linewidth=lw,
                linestyle=ls,
                zorder=2,
            )
        )

    def arrow(p, q, style="-|>", lw=0.7):
        ax.add_patch(
            FancyArrowPatch(
                p,
                q,
                arrowstyle=style,
                mutation_scale=5,
                linewidth=lw,
                color="black",
                shrinkA=0,
                shrinkB=0,
                zorder=3,
            )
        )

    # Student residual stream: removed depths are identity, the action tokens pass through unchanged.
    scx = se[0] + se[1] / 2
    ax.plot([scx, scx], [yb(0), yb(15) + rh], color=PALETTE["gray"], lw=0.7, zorder=1)
    for d in range(16):
        yc = yb(d) + rh / 2
        on = d in kept
        bar(*tv, d, vlm_c)
        bar(*te, d, act_c, ec=hid_c if on else edge, lw=0.9 if on else 0.5)
        ax.text(tv[0] - 0.04, yc, str(d), ha="right", va="center", fontsize=4.5, color=PALETTE["gray"])
        if on:
            j = kept.index(d)
            bar(*se, d, act_c, ec=hid_c, lw=0.9)
            bar(*sv, d, vlm_c)
            ax.text(scx, yc, f"layer {j}", ha="center", va="center", fontsize=4.5, zorder=3)
            ax.text(sv[0] + sv[1] / 2, yc, f"VLM {d}", ha="center", va="center", fontsize=4.5, zorder=3)
            # L_hid, --hidden_map same_depth: student layer j <-> teacher expert at the depth it was copied from.
            ax.plot([te[0] + te[1], se[0]], [yc, yc], color=hid_c, lw=0.7, zorder=1)
        else:
            bar(*sv, d, "white", ec=PALETTE["neutral"])
    ax.text(
        tv[0] - 0.2,
        (yb(0) + yb(15) + rh) / 2,
        "depth",
        rotation=90,
        ha="center",
        va="center",
        fontsize=5.5,
        color=PALETTE["gray"],
    )
    # Labels go in the empty rows of the gap (depths 6-7 and 2-3 have no L_hid line).
    ax.text(
        gx,
        yb(6) + pitch / 2 + rh / 2,
        "$\\mathcal{L}_{hid}$",
        ha="center",
        va="center",
        fontsize=8,
        color=hid_c,
    )
    ax.text(
        gx,
        yb(2) + pitch / 2 + rh / 2,
        "same depth:\nlayer copied from",
        ha="center",
        va="center",
        fontsize=4.5,
        color=hid_c,
        linespacing=1.15,
    )
    ax.text(
        scx,
        yb(2) + pitch / 2 + rh / 2,
        "identity",
        ha="center",
        va="center",
        fontsize=4.3,
        color=PALETTE["gray"],
        style="italic",
        rotation=90,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4},
        zorder=2,
    )

    # Outputs: velocity at the same (x_t, t) -> L_vel, and the 10-step action chunk -> L_out.
    yo = yb(15) + rh + 0.17
    tcx = te[0] + te[1] / 2
    for cx_, lab, side in [(tcx, "$v_\\phi$", -1), (scx, "$v_\\theta$", 1)]:
        arrow((cx_, yb(15) + rh), (cx_, yo - 0.04))
        for x in np.linspace(cx_ - 0.15, cx_ + 0.15, 4):
            ax.add_patch(Circle((x, yo), 0.03, facecolor=act_c, edgecolor=edge, linewidth=0.5, zorder=3))
        ax.text(cx_ + side * 0.2, yo, lab, ha="left" if side > 0 else "right", va="center", fontsize=7.5)
    arrow((tcx + 0.42, yo), (scx - 0.42, yo), style="<|-|>")
    ax.text(
        gx, yo + 0.03, "$\\mathcal{L}_{vel}$, $\\mathcal{L}_{out}$", ha="center", va="bottom", fontsize=7.5
    )

    # Inputs.
    yi = y0 - 0.1
    for x0, w, n, face in [
        (tv[0], tv[1], 5, vlm_c),
        (te[0], te[1], 4, act_c),
        (se[0], se[1], 4, act_c),
        (sv[0], sv[1], 5, vlm_c),
    ]:
        for i, x in enumerate(np.linspace(x0 + 0.06, x0 + w - 0.06, n)):
            if face is vlm_c and i == n - 2:
                ax.text(x, yi, "…", ha="center", va="center", fontsize=5)
            else:
                ax.add_patch(Circle((x, yi), 0.03, facecolor=face, edgecolor=edge, linewidth=0.5))
        ax.text(
            x0 + w / 2,
            yi - 0.055,
            "V-L tokens" if face is vlm_c else "action tokens",
            ha="center",
            va="top",
            fontsize=4.5,
        )
    ax.text(
        (tv[0] + te[0] + te[1]) / 2,
        0.36,
        "Teacher: 16 layers (frozen)",
        ha="center",
        va="center",
        fontsize=6.5,
    )
    ax.text(
        (se[0] + sv[0] + sv[1]) / 2,
        0.36,
        "Student: 8 layers (trained)",
        ha="center",
        va="center",
        fontsize=6.5,
        color=PALETTE["blue_main"],
    )
    ax.text(
        width_in / 2,
        0.12,
        "$\\mathcal{L} = \\mathcal{L}_{vel} + 0.1\\,\\mathcal{L}_{hid} + \\mathcal{L}_{out}$",
        ha="center",
        va="center",
        fontsize=7.5,
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_distill_diagram_paper.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"wrote {FIG / 'fig_distill_diagram_paper'}.{{pdf,png}}")


def plot_student_layouts() -> None:
    """Teacher and the three 8-layer student layouts (spread / early / late), compact full-width figure."""
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    frozen_c, frozen_e = "#E0E0E0", "#444444"
    trained_c, trained_e = "#A6CEE3", "#1F78B4"
    cw, ch = 0.82, 0.62  # cell width / height in data units (one depth = 1.0 in x)
    fs = 7

    def cell(x, y, kind):
        if kind == "frozen":
            kw = {"facecolor": frozen_c, "edgecolor": frozen_e, "linewidth": 0.8}
        elif kind == "trained":
            kw = {"facecolor": trained_c, "edgecolor": trained_e, "linewidth": 0.9}
        else:
            kw = {"facecolor": "white", "edgecolor": "#AAAAAA", "linewidth": 0.7, "linestyle": (0, (2, 1.5))}
        ax.add_patch(
            FancyBboxPatch((x - cw / 2, y - ch / 2), cw, ch, boxstyle="round,pad=0,rounding_size=0.08", **kw)
        )

    y_depth, y_attn, y_bb, y_t = 4.55, 4.15, 3.35, 2.2
    rows = [
        ("Student, spread", 1.05, [0, 1, 4, 5, 8, 9, 12, 13]),
        ("Student, early", 0.3, list(range(8))),
        ("Student, late", -0.45, list(range(8, 16))),
    ]
    lx = -0.7
    ax.text(lx, y_depth, "depth", ha="right", va="center", fontsize=6.5, color=PALETTE["gray"])
    ax.text(lx, y_attn, "attention", ha="right", va="center", fontsize=6.5, color=PALETTE["gray"])
    ax.text(lx, y_bb, "Backbone (VLM)", ha="right", va="center", fontsize=fs + 0.5)
    ax.text(lx, y_t, "Teacher expert", ha="right", va="center", fontsize=fs + 0.5)
    for i in range(16):
        ax.text(i, y_depth, str(i), ha="center", va="center", fontsize=6.5)
        ax.text(i, y_attn, "J" if i % 2 == 0 else "C", ha="center", va="center", fontsize=6.5, weight="bold")
        cell(i, y_bb, "frozen")
        cell(i, y_t, "frozen")
        ax.add_patch(
            FancyArrowPatch(
                (i, y_bb - ch / 2),
                (i, y_t + ch / 2),
                arrowstyle="-|>",
                mutation_scale=5,
                linewidth=0.6,
                color="#888888",
                shrinkA=0,
                shrinkB=0,
            )
        )
    for name, y, kept in rows:
        ax.text(lx, y, name, ha="right", va="center", fontsize=fs + 0.5)
        for i in range(16):
            cell(i, y, "trained" if i in kept else "removed")

    # Brace over depths 8-15: VLM layers the early student never reads.
    by, x0, x1, xm = 4.85, 7.6, 15.4, 11.5
    ax.plot(
        [x0, x0, xm - 0.15, xm, xm + 0.15, x1, x1],
        [by, by + 0.1, by + 0.1, by + 0.22, by + 0.1, by + 0.1, by],
        color="#666666",
        lw=0.8,
    )
    ax.text(
        xm,
        by + 0.3,
        "not needed by the early student, can be skipped",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color=PALETTE["dark"],
    )

    handles = [
        Patch(facecolor=frozen_c, edgecolor=frozen_e, label="frozen"),
        Patch(facecolor=trained_c, edgecolor=trained_e, label="trained, copied from the teacher"),
        Patch(facecolor="white", edgecolor="#AAAAAA", linestyle="--", label="removed (identity)"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(7.5, -0.95),
        bbox_transform=ax.transData,
        ncols=3,
        fontsize=fs,
        handlelength=1.4,
        handleheight=0.9,
        columnspacing=1.6,
        borderaxespad=0,
    )
    ax.set_xlim(-4.2, 15.6)
    ax.set_ylim(-1.75, 5.45)
    ax.set_axis_off()
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(
            FIG / f"fig_student_layouts.{ext}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
            facecolor="white",
        )
    plt.close(fig)
    print(f"wrote {FIG / 'fig_student_layouts'}.{{pdf,png}}")


# --------------------------------------------------------------------------------------------------
# Latency
# --------------------------------------------------------------------------------------------------


def plot_latency_breakdown() -> None:
    """Stacked bars per model and K: VLM prefix / expert non-FFN / expert FFN."""
    lat = latency()
    names = ["teacher", "k8_drop", "k8_fuse"]
    ks = ["1", "3", "10"]
    parts = [
        ("prefix_ms", "VLM prefix", PALETTE["neutral"], ""),
        ("non_ffn_ms", "Expert: attention + other", PALETTE["blue_secondary"], ""),
        ("ffn_ms", "Expert: FFN", PALETTE["highlight"], "//"),
    ]
    short = {"teacher": "Teacher", "k8_drop": "8L Alt", "k8_fuse": "8L Alt\nFFN Fuse"}
    fig, ax = plt.subplots(figsize=(11, 6.2))
    width, group_gap = 0.8, 1.0
    xticks, xlabels = [], []
    for g, k in enumerate(ks):
        for i, m in enumerate(names):
            x = g * (len(names) + group_gap) + i
            s = lat[m]["steps"][k]
            bottom = 0.0
            for key, _, color, hatch in parts:
                ax.bar(
                    x,
                    s[key],
                    width,
                    bottom=bottom,
                    color=color,
                    edgecolor="black",
                    linewidth=1.5,
                    hatch=hatch,
                )
                bottom += s[key]
            ax.text(x, s["wall_ms"] + 2, f"{s['wall_ms']:.0f}", ha="center", va="bottom", fontsize=13)
            xticks.append(x)
            xlabels.append(short[m])
        ax.text(
            g * (len(names) + group_gap) + 1,
            -17,
            f"K = {k}",
            ha="center",
            va="top",
            fontsize=17,
            weight="bold",
        )
    ax.set_xticks(xticks, xlabels, fontsize=13)
    for t, m in zip(ax.get_xticklabels(), names * len(ks), strict=True):
        t.set_color(MODELS[m][1] if m != "teacher" else "black")
    ax.set_ylabel("Latency per action chunk (ms) " + DOWN)
    ax.set_ylim(0, 135)
    ax.legend(
        handles=[Patch(facecolor=c, edgecolor="black", hatch=h, label=lab) for _, lab, c, h in parts][::-1],
        loc="upper left",
        fontsize=13,
    )
    ax.text(
        0.99, 0.98, GPU_NOTE, transform=ax.transAxes, ha="right", va="top", fontsize=11, color=PALETTE["gray"]
    )
    save(fig, "fig_latency_breakdown")


def plot_latency_vs_k() -> None:
    """Total chunk latency vs denoising steps K, with the line fit prefix + K * per_step."""
    lat = latency()
    fig, ax = plt.subplots(figsize=(8, 6))
    kk = np.linspace(0, 10.5, 50)
    for m in ["teacher", "k8_drop", "k8_fuse"]:
        label, color = MODELS[m]
        ks = sorted(int(k) for k in lat[m]["steps"])
        y = [lat[m]["steps"][str(k)]["wall_ms"] for k in ks]
        fit = lat[m]["fit"]
        ls = "--" if m == "k8_drop" else "-"
        ax.plot(kk, fit["prefix_ms"] + fit["per_step_ms"] * kk, color=color, lw=2.5, ls=ls, alpha=0.8)
        ax.plot(
            ks,
            y,
            "o",
            color=color,
            ms=10,
            markeredgecolor="black",
            markeredgewidth=1.2,
            label=f"{label}: {fit['per_step_ms']:.2f} ms / step",
        )
    prefix = lat["teacher"]["fit"]["prefix_ms"]
    ax.axhline(prefix, color="black", alpha=0.3, lw=3, ls=":")
    ax.text(
        10.4,
        prefix - 3,
        f"VLM prefix ≈ {prefix:.0f} ms",
        ha="right",
        va="top",
        fontsize=13,
        color=PALETTE["gray"],
    )
    ax.axvspan(2.7, 3.3, color=PALETTE["green_1"], zorder=0)
    ax.text(3, 3, "target\nK = 3", ha="center", va="bottom", fontsize=12, color=PALETTE["dark"])
    ax.set_xlabel("Denoising steps K")
    ax.set_ylabel("Latency per action chunk (ms) " + DOWN)
    ax.set_xticks([0, 1, 3, 5, 10])
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 135)
    ax.legend(loc="upper left", fontsize=12.5)
    ax.text(
        0.99,
        0.02,
        GPU_NOTE,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        color=PALETTE["gray"],
    )
    save(fig, "fig_latency_vs_k")


def plot_latency_3080() -> None:
    """Single-column (3.25 in) stacked bars: VLM prefix + action expert per chunk, teacher vs spread student,
    RTX 3080 Laptop with the SM clock locked at 1410 MHz."""
    models = json.loads((OUT / "latency" / "rtx3080_locked.json").read_text())["models"]
    names = [("teacher", "Teacher"), ("k8_pairs", "Spread")]
    ks = ["1", "3", "10"]
    parts = [
        ("prefix_ms", "VLM prefix", PALETTE["neutral"]),
        ("denoise_ms", "action expert (denoising)", PALETTE["blue_secondary"]),
    ]
    fs = 7
    fig, ax = plt.subplots(figsize=(3.25, 2.35))
    width, gap = 0.78, 0.7
    xticks, xlabels = [], []
    for g, k in enumerate(ks):
        for i, (m, label) in enumerate(names):
            x = g * (len(names) + gap) + i
            s = models[m]["steps"][k]
            bottom = 0.0
            for key, _, color in parts:
                ax.bar(x, s[key], width, bottom=bottom, color=color, edgecolor="black", linewidth=0.6)
                bottom += s[key]
            ax.text(x, bottom + 6, f"{s['wall_ms']:.0f}", ha="center", va="bottom", fontsize=fs - 0.5)
            xticks.append(x)
            xlabels.append(label)
        ax.text(
            g * (len(names) + gap) + 0.5,
            -45,
            f"K = {k}",
            ha="center",
            va="top",
            fontsize=fs + 0.5,
            weight="bold",
        )
    ax.set_xticks(xticks, xlabels, fontsize=fs - 1)
    for t in ax.get_xticklabels()[1::2]:
        t.set_color(PALETTE["blue_main"])
    ax.set_ylabel("Latency per action chunk (ms)", fontsize=fs + 0.5)
    ax.tick_params(axis="y", labelsize=fs, length=3, width=0.8)
    ax.tick_params(axis="x", length=0, pad=3)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.9)
    ax.set_ylim(0, 470)
    ax.set_xlim(-0.6, xticks[-1] + 0.6)
    ax.legend(
        handles=[Patch(facecolor=c, edgecolor="black", linewidth=0.6, label=lab) for _, lab, c in parts][
            ::-1
        ],
        loc="upper left",
        fontsize=fs,
        handlelength=1.2,
        handleheight=0.8,
        borderaxespad=0.2,
    )
    ax.text(
        -0.45,
        345,
        "Teacher: 16 expert layers\nSpread: 8 layers (0,1,4,5,8,9,12,13)",
        ha="left",
        va="top",
        fontsize=fs - 1.5,
        color=PALETTE["dark"],
        linespacing=1.3,
    )
    fig.text(
        0.99,
        0.01,
        "RTX 3080 Laptop, clock locked at 1410 MHz, batch 1",
        ha="right",
        va="bottom",
        fontsize=fs - 1.5,
        color=PALETTE["gray"],
    )
    fig.tight_layout(pad=0.3, rect=(0, 0.04, 1, 1))
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_latency_3080.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"wrote {FIG / 'fig_latency_3080'}.{{pdf,png}}")


def plot_ffn_width() -> None:
    """Single SwiGLU microbenchmark: cost grows sub-linearly in width at batch 1 (memory / launch bound)."""
    sweep = json.loads((OUT / "latency" / "breakdown_teacher.json").read_text())["ffn_width_sweep_us"]
    w = np.array(sorted(int(k) for k in sweep))
    us = np.array([sweep[str(k)] for k in w])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(w, us[0] * w / w[0], color="black", alpha=0.3, lw=3, ls="--", label="linear in width")
    ax.plot(w, us, "-o", color=PALETTE["blue_main"], lw=3, ms=11, markeredgecolor="black", label="measured")
    tags = {2048: "teacher", 4096: "k8 fuse", 8192: "k4 fuse"}
    for x, y in zip(w, us, strict=True):
        txt = f"{y / us[0]:.2f}×" + (f"\n{tags[x]}" if x in tags else "")
        first = x == w[0]
        ax.annotate(
            txt,
            (x, y),
            textcoords="offset points",
            xytext=(14, -10) if first else (0, -14),
            ha="left" if first else "center",
            va="top",
            fontsize=12.5,
        )
    ax.set_xticks(w)
    ax.set_xlabel("FFN intermediate width")
    ax.set_ylabel("Time per FFN call (µs) " + DOWN)
    ax.set_ylim(0, 185)
    ax.legend(loc="upper left", fontsize=13)
    ax.text(
        0.99,
        0.02,
        "one SwiGLU, hidden 720, 50 tokens, bf16\n" + GPU_NOTE,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        color=PALETTE["gray"],
    )
    save(fig, "fig_ffn_width")


def plot_cost() -> None:
    """Expert params, FLOP per step and latency (K = 10) of each student, relative to the teacher."""
    lat = latency()
    t = lat["teacher"]
    metrics = [
        ("Expert\nparameters", lambda m: m["expert_params"] / t["expert_params"]),
        ("FLOP per\ndenoising step", lambda m: m["flops"]["per_step"] / t["flops"]["per_step"]),
        (
            "Denoising\nlatency (K=10)",
            lambda m: m["steps"]["10"]["denoise_ms"] / t["steps"]["10"]["denoise_ms"],
        ),
        ("Chunk\nlatency (K=10)", lambda m: m["steps"]["10"]["wall_ms"] / t["steps"]["10"]["wall_ms"]),
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.36
    for i, m in enumerate(["k8_drop", "k8_fuse"]):
        label, color = MODELS[m]
        vals = [f(lat[m]) for _, f in metrics]
        x = np.arange(len(metrics)) + (i - 0.5) * width
        ax.bar(
            x,
            vals,
            width,
            color=color,
            edgecolor="black",
            linewidth=1.5,
            label=label,
            alpha=0.9 if m == "k8_fuse" else 1.0,
        )
        for xi, v in zip(x, vals, strict=True):
            ax.text(xi, v + 0.02, f"{v:.0%}", ha="center", va="bottom", fontsize=13)
    ax.axhline(1.0, color="black", alpha=0.3, lw=3, ls="--")
    ax.text(
        len(metrics) - 0.45,
        1.02,
        "teacher = 100%",
        ha="right",
        va="bottom",
        fontsize=12,
        color=PALETTE["gray"],
    )
    ax.set_xticks(np.arange(len(metrics)), [n for n, _ in metrics], fontsize=14)
    ax.set_ylabel("Relative to teacher")
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncols=2, fontsize=14)
    ax.text(
        0.99,
        -0.2,
        "latency: " + GPU_NOTE,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        color=PALETTE["gray"],
    )
    save(fig, "fig_cost_relative")


# --------------------------------------------------------------------------------------------------
# Distillation
# --------------------------------------------------------------------------------------------------


def plot_zeroshot() -> None:
    """Held-out chunk_mse_teacher of each student before training (step 0, 60-episode subset)."""

    def val(name):
        rs = read_jsonl(OUT / "zeroshot" / name / "metrics.jsonl")
        return next(r["chunk_mse_teacher"] for r in rs if r.get("split") == "val")

    keeps = [("alternate", ""), ("first", "_first"), ("last", "_last")]
    drop = [val(f"k8_drop{s}") for _, s in keeps]
    fuse = [val(f"k8_fuse{s}") for _, s in keeps]
    pairs = val("k8_pairs")
    fig, ax = plt.subplots(figsize=(9, 6))
    width = 0.36
    x = np.arange(len(keeps))
    for vals, m, off, zlabel in [
        (drop, "k8_drop", -0.5, "Shallow (dropped)"),
        (fuse, "k8_fuse", 0.5, "Shallow (FFN-fused)"),
    ]:
        _, color = MODELS[m]
        ax.bar(x + off * width, vals, width, color=color, edgecolor="black", linewidth=1.5, label=zlabel)
        for xi, v in zip(x + off * width, vals, strict=True):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=13)
    ax.axhline(
        pairs,
        color=PALETTE["green_3"],
        lw=3,
        ls="--",
        label=f"Shallow — 8L Spread ({pairs:.2f})",
        zorder=3,
    )
    ax.set_xticks(x, [f"keep {k}" for k, _ in keeps])
    ax.set_ylabel("Held-out chunk MSE vs teacher " + DOWN)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", fontsize=14, ncols=3, columnspacing=1.0, handlelength=1.6)
    ax.set_title("Before distillation (step 0)", fontsize=16)
    save(fig, "fig_zeroshot")


def _distill_runs() -> dict[str, list[dict]]:
    runs = {}
    for m in ["pairs", "k8_drop", "k8_fuse"]:
        p = OUT / "distill" / m / "metrics.jsonl"
        if p.exists():
            runs[m] = read_jsonl(p)
    return runs


def _plot_runs(ax, runs, split, key, smooth=None):
    for m, rs in runs.items():
        label, color = MODELS[m]
        pts = [(r["step"], r[key]) for r in rs if r.get("split") == split and key in r]
        pts = sorted(dict(pts).items())  # a resumed run logs some steps twice: keep the last
        x, y = np.array(pts).T
        if smooth:
            ax.plot(x, y, color=color, lw=1, alpha=0.2)
            y = np.array([y[max(0, i - smooth + 1) : i + 1].mean() for i in range(len(y))])
        ax.plot(
            x,
            y,
            color=color,
            lw=3 if m == "k8_fuse" else 2.5,
            marker=None if smooth else "o",
            ms=8,
            label=label,
            zorder=3 if m == "k8_fuse" else 2,
        )


def _log_ticks(ax) -> None:
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def plot_distill() -> None:
    """Held-out validation metrics vs training step, three runs."""
    runs = _distill_runs()
    if not runs:
        print("skip distill: no outputs/distill/<run>/metrics.jsonl")
        return
    last = min(max(r["step"] for r in rs if r.get("split") == "val") for rs in runs.values())
    panels = [
        ("chunk_mse_teacher", "Chunk MSE vs teacher ↓"),
        ("velocity", "Velocity loss ↓"),
        ("hidden", "Hidden-state loss ↓"),
        ("chunk_mse_gt", "Chunk MSE vs ground truth ↓"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    for ax, (key, title) in zip(axes, panels, strict=True):
        _plot_runs(ax, runs, "val", key)
        ax.set_title(title, fontsize=18)
        ax.set_xlabel("Training step")
        ax.set_yscale("log")
        _log_ticks(ax)
        if key == "chunk_mse_gt":
            ax.axhline(TEACHER_FLOOR, color="black", alpha=0.3, lw=3, ls="--")
            ax.text(
                ax.get_xlim()[1],
                TEACHER_FLOOR * 1.12,
                "teacher",
                ha="right",
                va="bottom",
                fontsize=13,
                color=PALETTE["gray"],
            )
    axes[0].legend(loc="upper right", fontsize=14)
    fig.suptitle(
        f"Held-out validation (85 episodes){'' if last >= 5000 else f', step ≤ {last}, runs in progress'}",
        fontsize=16,
        color=PALETTE["gray"],
        y=0.99,
    )
    save(fig, "fig_distill_val", pad=1.5)

    # Main panel alone, linear y, for the paper body.
    fig, ax = plt.subplots(figsize=(9, 6.5))
    _plot_runs(ax, runs, "val", "chunk_mse_teacher")
    for m, rs in runs.items():
        vals = [(r["chunk_mse_teacher"], r["step"]) for r in rs if r.get("split") == "val"]
        best, step = min(vals)
        ax.plot(step, best, marker="*", ms=22, color=MODELS[m][1], markeredgecolor="black", zorder=4)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Held-out chunk MSE vs teacher " + DOWN)
    ax.set_ylim(0, None)
    handles = [Line2D([0], [0], color=MODELS[m][1], lw=3, marker="o", ms=8, label=MODELS[m][0]) for m in runs]
    handles.append(
        Line2D(
            [0],
            [0],
            color="white",
            marker="*",
            ms=18,
            markerfacecolor=PALETTE["gray"],
            markeredgecolor="black",
            label="best checkpoint",
        )
    )
    ax.legend(handles=handles, loc="upper right", fontsize=14)
    if last < 5000:
        ax.text(
            0.99,
            0.6,
            f"in progress: step ≤ {last} of 5000",
            transform=ax.transAxes,
            ha="right",
            fontsize=12,
            color=PALETTE["gray"],
        )
    save(fig, "fig_distill_main")

    # Train losses (single batches, smoothed over 5 logs = 250 steps).
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    for ax, (key, title) in zip(
        axes,
        [
            ("total", "Total loss"),
            ("velocity", "Velocity loss"),
            ("hidden", "Hidden-state loss"),
            ("output", "Output (chunk) loss"),
        ],
        strict=True,
    ):
        _plot_runs(ax, runs, "train", key, smooth=5)
        ax.set_title(title + " (train) ↓", fontsize=18)
        ax.set_xlabel("Training step")
        ax.set_yscale("log")
        _log_ticks(ax)
    axes[0].legend(loc="upper right", fontsize=14)
    save(fig, "fig_distill_train", pad=1.5)


FIGURES = {
    "method_fusion": plot_method,
    "layouts": plot_layouts,
    "distill_diagram": plot_distill_diagram,
    "distill_diagram_small": plot_distill_diagram_small,
    "distill_diagram_paper": plot_distill_diagram_paper,
    "student_layouts": plot_student_layouts,
    "latency_breakdown": plot_latency_breakdown,
    "latency_vs_k": plot_latency_vs_k,
    "latency_3080": plot_latency_3080,
    "ffn_width": plot_ffn_width,
    "cost_relative": plot_cost,
    "zeroshot": plot_zeroshot,
    "distill": plot_distill,
}

if __name__ == "__main__":
    apply_style()
    wanted = sys.argv[1:]
    for name, fn in FIGURES.items():
        if not wanted or any(w in name for w in wanted):
            fn()
