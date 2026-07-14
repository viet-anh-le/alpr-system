import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ── Dữ liệu ─────────────────────────────────────────────
# 4 cột cộng dồn (novelty story) + 1 cột leave-one-out (backbone)
inc_labels = ["A0\nBaseline", "A1\n+Aug.", "A2\n+STN", "A5\nFull"]
inc_vals = [87.53, 90.12, 94.59, 95.01]
inc_x = [0, 1, 2, 3]
inc_delta = [None, "+2.59", "+4.47", "+0.42"]  # chênh so với cột trước

bb_x, bb_val = 4.5, 91.32  # A5 − backbone (đo bổ sung)
A5_val = 95.01

BLUE = "#1f77b4"  # khớp màu mặc định biểu đồ cũ
BLUE_LITE = "#9ecae1"  # cột backbone: nhạt + hatch = mã hoá phụ (an toàn CVD)

fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)

# ── Nhóm cột cộng dồn ───────────────────────────────────
bars = ax.bar(inc_x, inc_vals, width=0.62, color=BLUE, zorder=3)
for x, v, d in zip(inc_x, inc_vals, inc_delta):
    ax.text(x, v + 0.18, f"{v:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    if d:  # nhãn chênh lệch trong thân cột
        ax.text(x, v - 1.1, d + " pt", ha="center", va="top", fontsize=9.5, color="white")

# ── Cột leave-one-out (backbone) ────────────────────────
ax.bar(
    bb_x, bb_val, width=0.62, color=BLUE_LITE, edgecolor=BLUE, hatch="//", linewidth=1.2, zorder=3
)
ax.text(
    bb_x, bb_val + 0.18, f"{bb_val:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold"
)

# ── Vạch ngăn cách + mũi tên Δ +3.69pt ──────────────────
ax.axvline(3.9, color="#999", ls=":", lw=1.1, zorder=1)
ax.hlines(A5_val, 3, bb_x, color="#888", ls="--", lw=1, zorder=2)  # mốc A5
arrow = FancyArrowPatch(
    (bb_x, bb_val),
    (bb_x, A5_val),
    arrowstyle="<->",
    mutation_scale=12,
    color="#333",
    lw=1.3,
    zorder=4,
)
ax.add_patch(arrow)
ax.text(
    bb_x + 0.12,
    (bb_val + A5_val) / 2,
    "+3.69 pt",
    ha="left",
    va="center",
    fontsize=10.5,
    fontweight="bold",
)


# ── Trục & thẩm mỹ ──────────────────────────────────────
ax.set_xticks(inc_x + [bb_x])
ax.set_xticklabels(inc_labels + ["A5 − backbone\n(block LPRNet)"], fontsize=10)
ax.set_ylabel("Exact-match (%)", fontsize=11)
ax.set_title("Ablation study", fontsize=15, pad=12)
ax.set_ylim(84, 96.5)  # cắt trục y như biểu đồ gốc
ax.set_xlim(-0.7, 5.3)
ax.grid(axis="y", alpha=0.3, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("ablation_final.png", dpi=200, bbox_inches="tight")
plt.show()
