import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

FIG = Path("results/figures/fig_architecture.png")

fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")

def box(x, y, w, h, text, color):
    r = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                 linewidth=1.3, edgecolor="black", facecolor=color)
    ax.add_patch(r)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=9)

box(0.3, 5.5, 2.4, 1.0, "ICMR Anchors\n+ District Stats", "#cce5ff")
box(3.0, 5.5, 2.4, 1.0, "Synthetic AMR\n(643K records)", "#cce5ff")
box(5.7, 5.5, 2.4, 1.0, "Feature\nEngineering (52)", "#cce5ff")
box(8.4, 5.5, 3.0, 1.0, "District Graph\n(KNN + State)", "#cce5ff")

box(3.0, 3.7, 2.4, 1.0, "ST-GNN\n(GCN+GRU)", "#d4edda")
box(5.7, 3.7, 2.4, 1.0, "DML\n(Causal Layer)", "#d4edda")

box(1.5, 1.9, 2.4, 1.0, "Baselines\n(XGB / LSTM)", "#fff3cd")
box(4.2, 1.9, 2.4, 1.0, "Metrics\nR²/MAE/Recall", "#fff3cd")
box(6.9, 1.9, 2.4, 1.0, "Ablation\nStudy", "#fff3cd")
box(9.6, 1.9, 2.0, 1.0, "Figures\n+ Tables", "#fff3cd")

box(4.5, 0.2, 3.0, 1.0, "Paper + GitHub\n(Reproducible)", "#f8d7da")

for (x1, y1, x2, y2) in [
    (2.7, 6.0, 3.0, 6.0), (5.4, 6.0, 5.7, 6.0), (8.1, 6.0, 8.4, 6.0),
    (4.2, 5.5, 4.2, 4.7), (6.9, 5.5, 6.9, 4.7),
    (4.2, 3.7, 2.7, 2.9), (4.2, 3.7, 5.4, 2.9),
    (6.9, 3.7, 8.1, 2.9), (6.9, 3.7, 10.6, 2.9),
    (5.4, 1.9, 6.0, 1.2), (8.1, 1.9, 6.0, 1.2),
]:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.1))

ax.set_title("ResistNet-ST System Architecture", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG, dpi=150)
plt.close()
print("Saved:", FIG)