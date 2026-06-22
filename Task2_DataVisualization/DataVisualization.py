# ============================================================
#  TASK 3: Data Visualization — Titanic Dataset
#  Tools: Matplotlib + Seaborn
#  Goal : Transform raw data into visual formats,
#         reveal insights clearly, craft a data story
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Palette & Style ──────────────────────────────────────────
BG       = "#0F1117"
CARD     = "#1A1D27"
ACCENT1  = "#4CC9F0"   # cyan  – survived
ACCENT2  = "#F72585"   # pink  – did not survive
ACCENT3  = "#7209B7"   # purple
ACCENT4  = "#F4A261"   # amber
TEXT     = "#E8EAF0"
SUBTEXT  = "#8B90A7"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    CARD,
    "axes.edgecolor":    "#2A2D3E",
    "axes.labelcolor":   TEXT,
    "xtick.color":       SUBTEXT,
    "ytick.color":       SUBTEXT,
    "text.color":        TEXT,
    "grid.color":        "#2A2D3E",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "axes.grid":         True,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  False,
    "axes.spines.bottom":False,
})

# ── Load Data ────────────────────────────────────────────────
url = ("https://raw.githubusercontent.com/datasciencedojo/"
       "datasets/master/titanic.csv")
df = pd.read_csv(url)

# Prep
df["AgeGroup"] = pd.cut(df["Age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=["Child\n(0-12)", "Teen\n(13-18)",
            "Young Adult\n(19-35)", "Adult\n(36-60)", "Senior\n(60+)"])
df["FareBand"] = pd.qcut(df["Fare"], 4,
    labels=["Low", "Medium", "High", "Very High"])

# ============================================================
#  FIGURE: Full Dashboard (3×3 grid)
# ============================================================
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor(BG)

gs = gridspec.GridSpec(3, 3, figure=fig,
                       hspace=0.55, wspace=0.38,
                       top=0.91, bottom=0.06,
                       left=0.06, right=0.97)

# ── Header ───────────────────────────────────────────────────
fig.text(0.5, 0.965, "TITANIC  ·  DATA VISUALIZATION STORY",
         ha="center", fontsize=22, fontweight="bold",
         color=TEXT)
fig.text(0.5, 0.945,
         "A visual exploration of survival patterns across gender, class, age, and fare",
         ha="center", fontsize=11, color=SUBTEXT)

# helper: axis title
def ax_title(ax, title, sub=""):
    ax.set_title(title, fontsize=12, fontweight="bold",
                 color=TEXT, pad=10, loc="left")
    if sub:
        ax.text(0, 1.01, sub, transform=ax.transAxes,
                fontsize=8, color=SUBTEXT)

# ── CHART 1: Donut — Overall Survival ───────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(CARD)
surv = df["Survived"].value_counts()
wedges, _ = ax1.pie(
    surv.values, startangle=90,
    colors=[ACCENT2, ACCENT1],
    wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=3),
)
pct = surv[1] / surv.sum() * 100
ax1.text(0, 0.08, f"{pct:.1f}%", ha="center", fontsize=22,
         fontweight="bold", color=ACCENT1)
ax1.text(0, -0.22, "survived", ha="center", fontsize=10, color=SUBTEXT)
legend_patches = [
    mpatches.Patch(color=ACCENT1, label=f"Survived ({surv[1]})"),
    mpatches.Patch(color=ACCENT2, label=f"Did Not ({surv[0]})"),
]
ax1.legend(handles=legend_patches, loc="lower center",
           bbox_to_anchor=(0.5, -0.15), ncol=2,
           fontsize=8, frameon=False, labelcolor=TEXT)
ax_title(ax1, "Overall Survival")

# ── CHART 2: Grouped Bar — Survival by Gender ───────────────
ax2 = fig.add_subplot(gs[0, 1])
gender_data = df.groupby(["Sex", "Survived"]).size().unstack()
x = np.arange(2)
w = 0.35
b1 = ax2.bar(x - w/2, gender_data[0], w, color=ACCENT2,
             alpha=0.85, label="Did Not Survive", zorder=3)
b2 = ax2.bar(x + w/2, gender_data[1], w, color=ACCENT1,
             alpha=0.85, label="Survived", zorder=3)
for bar in list(b1) + list(b2):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 4,
             str(int(bar.get_height())),
             ha="center", fontsize=9, color=TEXT, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(["Female", "Male"], fontsize=10)
ax2.set_ylabel("Passengers", color=SUBTEXT, fontsize=9)
ax2.legend(fontsize=8, frameon=False, labelcolor=TEXT)
ax2.set_ylim(0, gender_data.max().max() * 1.18)
ax_title(ax2, "Survival by Gender")

# ── CHART 3: Horizontal Bar — Survival Rate by Class ────────
ax3 = fig.add_subplot(gs[0, 2])
pc = df.groupby("Pclass")["Survived"].mean() * 100
classes = ["1st Class", "2nd Class", "3rd Class"]
colors_bar = [ACCENT1, ACCENT4, ACCENT2]
bars = ax3.barh(classes, pc.values, color=colors_bar,
                alpha=0.85, height=0.5, zorder=3)
for bar, val in zip(bars, pc.values):
    ax3.text(val + 0.8, bar.get_y() + bar.get_height()/2,
             f"{val:.1f}%", va="center", fontsize=10,
             color=TEXT, fontweight="bold")
ax3.set_xlim(0, 100)
ax3.set_xlabel("Survival Rate (%)", color=SUBTEXT, fontsize=9)
ax3.invert_yaxis()
ax_title(ax3, "Survival Rate by Class")

# ── CHART 4: Overlapping Histogram — Age Distribution ───────
ax4 = fig.add_subplot(gs[1, :2])
s  = df[df["Survived"]==1]["Age"].dropna()
ns = df[df["Survived"]==0]["Age"].dropna()
ax4.hist(ns, bins=32, color=ACCENT2, alpha=0.6,
         label="Did Not Survive", zorder=3)
ax4.hist(s,  bins=32, color=ACCENT1, alpha=0.6,
         label="Survived", zorder=3)
ax4.axvline(s.median(),  color=ACCENT1, lw=2, ls="--",
            label=f"Survivor median: {s.median():.0f}")
ax4.axvline(ns.median(), color=ACCENT2, lw=2, ls="--",
            label=f"Non-survivor median: {ns.median():.0f}")
ax4.fill_betweenx([0, ax4.get_ylim()[1] if ax4.get_ylim()[1]>0 else 60],
                  0, 16, color=ACCENT1, alpha=0.05)
ax4.text(8, 1, "Children\n(0-16)", ha="center",
         fontsize=8, color=ACCENT1, alpha=0.7)
ax4.set_xlabel("Age", color=SUBTEXT, fontsize=9)
ax4.set_ylabel("Count", color=SUBTEXT, fontsize=9)
ax4.legend(fontsize=8, frameon=False, labelcolor=TEXT)
ax_title(ax4, "Age Distribution  —  Survivors vs Non-Survivors",
         "Shaded area = children (age < 16)")

# ── CHART 5: Heatmap — Gender × Class ───────────────────────
ax5 = fig.add_subplot(gs[1, 2])
pivot = df.pivot_table("Survived", index="Sex",
                       columns="Pclass", aggfunc="mean") * 100
sns.heatmap(pivot, annot=True, fmt=".1f",
            cmap=sns.light_palette(ACCENT1, as_cmap=True),
            linewidths=2, linecolor=BG,
            ax=ax5, cbar=False,
            annot_kws={"fontsize": 13, "fontweight": "bold",
                       "color": "#0F1117"})
ax5.set_xticklabels(["1st", "2nd", "3rd"], fontsize=10)
ax5.set_yticklabels(["Female", "Male"], rotation=0, fontsize=10)
ax5.set_xlabel("Class", color=SUBTEXT, fontsize=9)
ax5.set_ylabel("", color=SUBTEXT)
ax_title(ax5, "Survival % : Gender × Class")

# ── CHART 6: Violin — Age by Class & Survival ───────────────
ax6 = fig.add_subplot(gs[2, :2])
df_v = df.dropna(subset=["Age"])
df_v["Class"] = df_v["Pclass"].map({1:"1st",2:"2nd",3:"3rd"})
df_v["Status"] = df_v["Survived"].map({1:"Survived",0:"Did Not"})
sns.violinplot(data=df_v, x="Class", y="Age", hue="Status",
               split=True, palette={"Survived": ACCENT1, "Did Not": ACCENT2},
               inner="quartile", linewidth=1.2, ax=ax6)
ax6.set_xlabel("Passenger Class", color=SUBTEXT, fontsize=9)
ax6.set_ylabel("Age", color=SUBTEXT, fontsize=9)
legend = ax6.get_legend()
if legend:
    legend.set_frame_on(False)
    for text in legend.get_texts():
        text.set_color(TEXT)
ax_title(ax6, "Age Distribution by Class & Survival (Violin)",
         "Inner lines show quartiles")

# ── CHART 7: Stacked Bar — Fare Band vs Survival ────────────
ax7 = fig.add_subplot(gs[2, 2])
fare_data = df.groupby(["FareBand", "Survived"]).size().unstack(fill_value=0)
fare_data_pct = fare_data.div(fare_data.sum(axis=1), axis=0) * 100
bottom = np.zeros(len(fare_data_pct))
for surv_val, color, label in [(0, ACCENT2, "Did Not"), (1, ACCENT1, "Survived")]:
    vals = fare_data_pct[surv_val].values
    ax7.bar(fare_data_pct.index, vals, bottom=bottom,
            color=color, alpha=0.85, label=label, zorder=3)
    for i, (v, b) in enumerate(zip(vals, bottom)):
        if v > 8:
            ax7.text(i, b + v/2, f"{v:.0f}%",
                     ha="center", va="center",
                     fontsize=9, color="#0F1117", fontweight="bold")
    bottom += vals
ax7.set_ylim(0, 100)
ax7.set_ylabel("Proportion (%)", color=SUBTEXT, fontsize=9)
ax7.set_xlabel("Fare Band", color=SUBTEXT, fontsize=9)
ax7.legend(fontsize=8, frameon=False, labelcolor=TEXT)
ax_title(ax7, "Survival % by Fare Band")

# ── Footer ───────────────────────────────────────────────────
fig.text(0.5, 0.02,
         "Data Source: Titanic Dataset  •  Tools: Python · Matplotlib · Seaborn  •  Task 3: Data Visualization",
         ha="center", fontsize=8, color=SUBTEXT)

# ── Save ─────────────────────────────────────────────────────
out = "/mnt/user-data/outputs/titanic_task3_visualization.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"✅ Dashboard saved → {out}")
