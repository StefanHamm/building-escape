import os
import numpy as np
import matplotlib.pyplot as plt
import csv 

from loader import loadFloorPlan
import robustness
from visualize import plot_heatmap
from robustness import run_spatial_heatmap

base = os.path.join(os.path.dirname(__file__), '..')
floor_file = os.path.join(base, 'data', 'floorPlans', 'small.fplan')
sff_file = os.path.join(base, 'data', 'floorPlansSSF', 'small_sff.npy')

out_dir = os.path.join(os.path.dirname(__file__), "results", "robustness")
os.makedirs(out_dir, exist_ok=True)

# parameters
agent_count = 15
k = 0.1
xi = 0.05
n_runs = 200
max_steps = 5000

# load
floor = loadFloorPlan(floor_file)
sff = np.load(sff_file)

# run monte-carlo (uses robustness.run_monte_carlo)
res = robustness.run_monte_carlo(floor, sff, agent_count=agent_count, k=k, xi=xi, n_runs=n_runs, max_steps=max_steps)
times = np.asarray(res["all_times"])

print(f"mean={res['mean_time']:.2f}, std={res['std_time']:.2f}, min={res['min_time']}, max={res['max_time']}")

# 1) Histogram of evacuation times
plt.figure(figsize=(6,4))
plt.hist(times, bins=30, color="C0", edgecolor="k")
plt.xlabel("Evacuation time (steps)")
plt.ylabel("Count")
plt.title(f"Evacuation time distribution (k={k}, xi={xi}, n={n_runs})")
plt.tight_layout()
hist_path = os.path.join(out_dir, f"evac_hist_k{str(k).replace('.','p')}_xi{str(xi).replace('.','p')}.png")
plt.savefig(hist_path)
plt.close()
print(f"Saved histogram to {hist_path}")

# 2) Survival curve fraction NOT evacuated by time t
t_max = int(times.max())
ts = np.arange(0, t_max + 1)
surv = np.array([(times > t).mean() for t in ts])

plt.figure(figsize=(6,4))
plt.plot(ts, surv, drawstyle="steps-post")
plt.xlabel("Time (steps)")
plt.ylabel("Fraction not evacuated")
plt.title(f"Survival curve (k={k}, xi={xi})")
plt.ylim(-0.02, 1.02)
plt.tight_layout()
surv_path = os.path.join(out_dir, f"survival_k{str(k).replace('.','p')}_xi{str(xi).replace('.','p')}.png")
plt.savefig(surv_path)
plt.close()


print(f"Saved survival curve to {surv_path}")



#3)Spatial congestion heatmap


heat = run_spatial_heatmap(
    floor_layout=floor,
    layout_sff=sff,
    agent_count=agent_count,
    k=k,
    xi=xi,
    n_runs=200,
    max_steps=max_steps
)
# --------------------------------------------------
# Build masks from character-encoded floor plan
# --------------------------------------------------

# Free space = 'F'
# Exit       = 'E'
# Wall       = everything else
wall_mask = (floor != 'F') & (floor != 'E')
exit_mask = (floor == 'E')

# Mask walls so they are not colored
heat_masked = heat.copy()
heat_masked[wall_mask] = np.nan

# --------------------------------------------------
# Plot
# --------------------------------------------------

plt.figure(figsize=(6, 6))

im = plt.imshow(
    heat_masked,
    cmap="viridis",
    origin="lower"
)

out_path = os.path.join(out_dir, "spatial_congestion_heatmap.png")

# Overlay walls (black squares)
wy, wx = np.where(wall_mask)
plt.scatter(wx, wy, c="black", s=40, marker="s", label="Wall")

# Overlay exits (red squares)
ey, ex = np.where(exit_mask)
plt.scatter(ex, ey, c="red", s=60, marker="s", label="Exit")
# plot_heatmap(
#     heat,
#     xlabel="y",
#     ylabel="x",
#     title="Spatial congestion heatmap (Monte-Carlo averaged)",
#     out_path=out_path
# )

# print(f"Saved spatial heatmap to {out_path}")
plt.xlabel("y")
plt.ylabel("x")
plt.title("Spatial congestion heatmap (Monte-Carlo averaged)")
# plt.legend(
#     loc="lower right",
#     bbox_to_anchor=(1.02, 0.5),
#     fontsize=8,
#     frameon=False
# )
plt.colorbar(shrink=0.8)
plt.tight_layout()

# Save figure
heatmap_path = os.path.join(out_dir, "spatial_congestion_heatmap.png")
plt.savefig(heatmap_path)
plt.close()

print(f"Saved spatial congestion heatmap to {heatmap_path}")


# # grid for heatmap (reasonable default)
# ks = [0.05, 0.1, 0.2, 0.5, 1.0]           # coupling / sensitivity values (coarse)
# xis = [0.0, 0.05, 0.1, 0.3, 0.5]         # friction / noise values (coarse)

# # for finer resolution, increase length of ks / xis but runtime grows fast.
# # ks = [0.01, 0.03, 0.1, 0.3, 1.0]
# # xis = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3]

# mat = np.full((len(ks), len(xis)), np.nan, dtype=float)
# for i, k in enumerate(ks):
#     for j, xi in enumerate(xis):
#         print(f"Monte-Carlo sweep: k={k}, xi={xi}")
#         try:
#             res = robustness.run_monte_carlo(
#                 floor_layout=floor,
#                 layout_sff=sff,
#                 agent_count=agent_count,
#                 k=k,
#                 xi=xi,
#                 n_runs=n_runs,
#                 max_steps=max_steps
#             )
#             mat[i, j] = float(res["mean_time"])
#         except Exception as e:
#             print(f"  ERROR for k={k}, xi={xi}: {e}")
#             mat[i, j] = np.nan

# # save numeric results
# np.save(os.path.join(out_dir, f"mean_evac_matrix_ac{agent_count}.npy"), mat)
# csv_path = os.path.join(out_dir, f"mean_evac_matrix_ac{agent_count}.csv")
# with open(csv_path, "w", newline="") as fh:
#     writer = csv.writer(fh)
#     writer.writerow(["k/xi"] + [str(x) for x in xis])
#     for i, k in enumerate(ks):
#         writer.writerow([str(k)] + [("" if np.isnan(v) else f"{v:.3f}") for v in mat[i, :]])

# # plot heatmap (uses your visualize.plot_heatmap)
# out_heatmap = os.path.join(out_dir, f"mean_evacuation_time_heatmap_ac{agent_count}.png")
# plot_heatmap(mat, xvals=xis, yvals=ks, xlabel="xi", ylabel="k",
#              title=f"Mean evacuation time (ac={agent_count})", out_path=out_heatmap)
# print(f"Saved heatmap to {out_heatmap}")


# print(f"Saved survival curve to {surv_path}")