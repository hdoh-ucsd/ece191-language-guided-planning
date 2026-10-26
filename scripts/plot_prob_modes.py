"""Example figure: how MTR binds a probability to each candidate future trajectory.

Left: the candidates in the scene (world frame, metres), colour and line width scaled by
probability, lanes from the model's own map input for context.
Right: the same probabilities as a bar chart in the same colours.
"""
import glob, pickle, sys
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

CACHE, OUT_DIR, PRED_FILE, PNG = sys.argv[1:5]

# --- palette (reference instance from the dataviz skill) -------------------------------
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]   # blue 250 -> 700
cmap = LinearSegmentedColormap.from_list("blue_seq", RAMP)

# --- data -------------------------------------------------------------------------------
pred = torch.load(f"{OUT_DIR}/{PRED_FILE}", map_location="cpu", weights_only=False)
scen, obj, t_idx = PRED_FILE.split("___")[:3]
trajs, probs = np.asarray(pred["pred_trajs"]), np.asarray(pred["pred_scores"])
gt = np.asarray(pred["gt_trajs"]); past, fut = gt[:21], gt[21:81]
past, fut = past[past[:, -1] > 0], fut[fut[:, -1] > 0]

# map polylines of the matching preprocessed sample (target-centred) -> world frame
sample = None
for chunk in sorted(glob.glob(f"{CACHE}/sd_nuplan_v1.1_{scen}.pkl___[0-9]*.pkl")):
    for s in pickle.load(open(chunk, "rb")):
        if str(s["center_objects_id"]) == obj and int(s["current_time_indices"]) == int(t_idx):
            sample = s
    if sample is not None:
        break
c = sample["center_objects_world"]; cx, cy, yaw = float(c[0]), float(c[1]), float(c[6])
R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
mc = np.asarray(sample.get("map_center", np.zeros(3)))[:2]
lanes = []
for poly, mask in zip(sample["map_polylines"], sample["map_polylines_mask"]):
    pts = poly[mask][:, :2]
    if len(pts) >= 2:
        lanes.append(pts @ R.T + np.array([cx, cy]) + mc)

# --- figure -----------------------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED})
fig = plt.figure(figsize=(14, 7.2), facecolor=SURFACE)
gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], wspace=0.22, left=0.05, right=0.97, top=0.84, bottom=0.11)
ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
for a in (ax, bx):
    a.set_facecolor(SURFACE)
    for sp in a.spines.values():
        sp.set_color(AXIS); sp.set_linewidth(0.8)

pmax = probs.max()
order = np.argsort(-probs)            # already sorted, but make the rank explicit
rank = {m: r + 1 for r, m in enumerate(order)}

# lanes (recessive)
for pts in lanes:
    ax.plot(pts[:, 0], pts[:, 1], color=GRID, lw=0.9, zorder=1, solid_capstyle="round")
# candidates, lowest probability first so the likely ones draw on top
for m in order[::-1]:
    p = probs[m]; col = cmap(p / pmax); lw = 1.2 + 2.6 * p / pmax
    ax.plot(trajs[m, :, 0], trajs[m, :, 1], color=col, lw=lw, zorder=3, solid_capstyle="round")
    ax.plot(trajs[m, -1, 0], trajs[m, -1, 1], "o", ms=8 + 4 * p / pmax, mfc=col, mec=SURFACE, mew=2, zorder=4)
# history and recorded future
ax.plot(past[:, 0], past[:, 1], color=INK, lw=2.6, zorder=5, solid_capstyle="round")
ax.plot(fut[:, 0], fut[:, 1], color=INK, lw=1.8, ls=(0, (3, 2.5)), zorder=8)
ax.plot(fut[-1, 0], fut[-1, 1], "D", ms=8, mfc=INK, mec=SURFACE, mew=2, zorder=8)
ax.annotate(f"actual at +{len(fut) / 10:.0f} s", xy=fut[-1, :2], xytext=(10, 6), textcoords="offset points",
            color=INK, fontsize=9, zorder=8)
ax.plot(past[-1, 0], past[-1, 1], "o", ms=9, mfc=INK, mec=SURFACE, mew=2, zorder=6)
# direct labels on the three most likely modes
for m in order[:3]:
    e = trajs[m, -1]; d = e - trajs[m, -8]; d = d / (np.linalg.norm(d) + 1e-6) * 3.0
    ax.annotate(f"#{rank[m]}  p = {probs[m]:.2f}", xy=e, xytext=e + d, color=INK, fontsize=10, fontweight="bold",
                ha="center", va="center", zorder=7,
                bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec=AXIS, lw=0.8))
ax.annotate("now", xy=past[-1, :2], xytext=(-14, -14), textcoords="offset points", color=INK2, fontsize=9, zorder=7)

allpts = np.vstack([trajs.reshape(-1, 2), past[:, :2], fut[:, :2]])
lo, hi = allpts.min(0) - 8, allpts.max(0) + 8
span = max(hi - lo); mid = (lo + hi) / 2
ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2); ax.set_ylim(mid[1] - span / 2, mid[1] + span / 2)
ax.set_aspect("equal"); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_xlabel("x (m, scenario frame)"); ax.set_ylabel("y (m)")
ax.set_title("Where each candidate goes in the next 6 s", loc="left", color=INK, fontsize=12, pad=10)
ax.legend(handles=[Line2D([], [], color=INK, lw=2.6, label="observed history (2 s)"),
                   Line2D([], [], color=INK, lw=1.8, ls=(0, (3, 2.5)), marker="D", mfc=INK, mec=SURFACE, ms=7,
                          label="recorded future (not seen by the model)"),
                   Line2D([], [], color=RAMP[2], lw=2.4, marker="o", mfc=RAMP[2], mec=SURFACE, mew=1.5, ms=8,
                          label="candidate trajectory: darker + thicker = more probable"),
                   Line2D([], [], color=GRID, lw=1.2, label="lane centrelines given to the model")],
          loc="lower left", frameon=True, facecolor=SURFACE, edgecolor=AXIS, fontsize=9)

# bar chart of the same probabilities
ranks = np.arange(1, len(order) + 1)
cols = [cmap(probs[m] / pmax) for m in order]
bx.barh(ranks, probs[order], color=cols, height=0.72, zorder=3)
for r, m in zip(ranks, order):
    bx.text(probs[m] + 0.004, r, f"{probs[m]:.2f}", va="center", color=INK2, fontsize=9.5, zorder=4)
bx.invert_yaxis(); bx.set_yticks(ranks); bx.set_yticklabels([f"#{r}" for r in ranks])
bx.set_xlim(0, pmax * 1.32); bx.set_xlabel("probability of this candidate")
bx.grid(axis="x", color=GRID, lw=0.6, zorder=0); bx.set_axisbelow(True)
for side in ("top", "right"): bx.spines[side].set_visible(False)
bx.set_title("The probability bound to each candidate", loc="left", color=INK, fontsize=12, pad=10)
kept = probs.sum()
bx.text(0.0, len(order) + 1.15, f"sum of the {len(order)} kept candidates = {kept:.2f};  the other {1 - kept:.2f} sits on\n"
        f"{64 - len(order)} candidates pruned as near-duplicates (< 2.5 m apart at 6 s)",
        color=INK2, fontsize=9, va="top")
bx.set_ylim(len(order) + 2.2, 0.3)

fig.suptitle("One MTR forecast = a set of candidate futures, each carrying its own probability", x=0.05, ha="left",
             fontsize=14, fontweight="bold", color=INK, y=0.965)
fig.text(0.05, 0.905, f"nuPlan scenario {scen}, vehicle {obj}, predicted at frame {t_idx}.  Each candidate is 60 (x, y) "
         "waypoints 0.1 s apart. The dashed line is what the car actually did.", color=INK2, fontsize=10)
fig.savefig(PNG, dpi=170, facecolor=SURFACE)
print("saved", PNG, "| probs:", probs.round(3).tolist(), "| lanes drawn:", len(lanes))
