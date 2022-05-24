import os, sys

import numpy as np
import matplotlib.pyplot as plt

mao_9_labels = ["No Norm Clipping", "Norm Clipping >= 0.4", "Norm Clipping >= 0.7"]
mao_3_labels = ["No Norm Clipping", "Norm Clipping >= 0.4", "Norm Clipping >= 0.7"]

mao_9_sizes = [224842, 119600, 35585]
mao_3_sizes = [231286, 131811, 42478]

mao_9_acc       = [54.70, 62.93, 66.72]
mao_9_acc_top2  = [67.19, 75.17, 81.53]
mao_9_acc_top3  = [72.92, 80.20, 86.07]
mao_9_acc_top5  = [79.25, 85.13, 90.89]
mao_9_acc_top10 = [85.64, 89.11, 94.32]

mao_3_acc       = [53.81, 60.53, 68.38]
mao_3_acc_top2  = [65.56, 72.64, 82.25]
mao_3_acc_top3  = [70.90, 77.76, 86.67]
mao_3_acc_top5  = [76.75, 83.16, 90.71]
mao_3_acc_top10 = [82.74, 87.73, 94.13]

fig, ax = plt.subplots()

# MAO 9
plt.vlines(x=mao_9_sizes[0], ymin=mao_9_acc[0], ymax=mao_9_acc_top10[0]-1.5, linestyle="-", linewidth=1.5, color="#e80911")
plt.vlines(x=mao_9_sizes[1], ymin=mao_9_acc[1], ymax=mao_9_acc_top10[1]-1.5, linestyle="--", linewidth=1.5, color="#e80911")
plt.vlines(x=mao_9_sizes[2], ymin=mao_9_acc[2], ymax=mao_9_acc_top10[2]-1.5, linestyle=":", linewidth=1.5, color="#e80911")

plt.plot(mao_9_sizes, mao_9_acc, color='black', linewidth=0, marker='o', markerfacecolor='#e80911', markeredgewidth=3, markersize=9)
plt.plot(mao_9_sizes, mao_9_acc_top2, color='black', linewidth=0, marker='v', markerfacecolor='#e80911', markeredgewidth=3, markersize=9)
plt.plot(mao_9_sizes, mao_9_acc_top3, color='black', linewidth=0, marker='s', markerfacecolor='#e80911', markeredgewidth=3, markersize=9)
plt.plot(mao_9_sizes, mao_9_acc_top5, color='black', linewidth=0, marker='p', markerfacecolor='#e80911', markeredgewidth=3, markersize=9)
plt.plot(mao_9_sizes, mao_9_acc_top10, color='#e80911', linewidth=0, marker='$10$', markerfacecolor='#e80911', markeredgewidth=1, markersize=13)


# MAO 3
plt.vlines(x=mao_3_sizes[0], ymin=mao_3_acc[0], ymax=mao_3_acc_top10[0]-1.5, linestyle="-", linewidth=1.5, color="#00a2ff")
plt.vlines(x=mao_3_sizes[1], ymin=mao_3_acc[1], ymax=mao_3_acc_top10[1]-1.5, linestyle="--", linewidth=1.5, color="#00a2ff")
plt.vlines(x=mao_3_sizes[2], ymin=mao_3_acc[2], ymax=mao_3_acc_top10[2]-1.5, linestyle=":", linewidth=1.5, color="#00a2ff")

plt.plot(mao_3_sizes, mao_3_acc, color='black', linewidth=0, marker='o', markerfacecolor='#00a2ff', markeredgewidth=3, markersize=9)
plt.plot(mao_3_sizes, mao_3_acc_top2, color='black', linewidth=0, marker='v', markerfacecolor='#00a2ff', markeredgewidth=3, markersize=9)
plt.plot(mao_3_sizes, mao_3_acc_top3, color='black', linewidth=0, marker='s', markerfacecolor='#00a2ff', markeredgewidth=3, markersize=9)
plt.plot(mao_3_sizes, mao_3_acc_top5, color='black', linewidth=0, marker='p', markerfacecolor='#00a2ff', markeredgewidth=3, markersize=9)
plt.plot(mao_3_sizes, mao_3_acc_top10, color='#00a2ff', linewidth=0, marker='$10$', markerfacecolor='#00a2ff', markeredgewidth=1, markersize=13)


plt.title('Accuracy vs Dataset Size (A-vs-C Loss)', fontweight="bold")
ax.set_xlim(0, 250000)
ax.set_ylim(top=100.0)
plt.xlabel("Dataset Size (Number of QA pairs)", fontweight="bold")
plt.ylabel('Accuracy (%)', fontweight="bold")
plt.xticks(fontweight="bold")
plt.yticks(fontweight="bold")
legends = [
    plt.Line2D([0], [0], color="black", linestyle=":", lw=1.5),
    plt.Line2D([0], [0], color="black", linestyle="--", lw=1.5),
    plt.Line2D([0], [0], color="black", linestyle="-", lw=1.5),
    plt.Line2D([0], [0], color="black", lw=0, marker='o', markerfacecolor="#e80911", markeredgewidth=3, markersize=9),
    plt.Line2D([0], [0], color="black", lw=0, marker='o', markerfacecolor="#00a2ff", markeredgewidth=3, markersize=9),
]
plt.legend(legends, ["Norm Clipping >= 0.7", "Norm Clipping >= 0.4", "No Norm Clipping", "Min Ans Occ >= 9", "Min Ans Occ >= 3"], ncol=2)

# The spines
plt.setp(ax.spines.values(), linewidth=3)

# The ticks
ax.xaxis.set_tick_params(width=3)
ax.yaxis.set_tick_params(width=3)
plt.grid(axis="y")

plt.setp(plt.gca().get_xticklabels(), rotation=0, horizontalalignment='right')

plt.show()
