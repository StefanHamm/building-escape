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
k = 5
xi = 0.5
n_runs = 200
max_steps = 5000


floor = loadFloorPlan(floor_file)
sff = np.load(sff_file)


res = robustness.run_monte_carlo(floor, sff, agent_count=agent_count, k=k, xi=xi, n_runs=n_runs, max_steps=max_steps)
times = np.asarray(res["all_times"])

print(f"mean={res['mean_time']:.2f}, std={res['std_time']:.2f}, min={res['min_time']}, max={res['max_time']}, failure_rate={res['failure_rate']:.2%}")

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
# Create masks for walls and exits

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
    origin="upper"
)

out_path = os.path.join(out_dir, "spatial_congestion_heatmap.png")


wy, wx = np.where(wall_mask)
plt.scatter(wx, wy, c="black", s=40, marker="s", label="Wall")


ey, ex = np.where(exit_mask)
plt.scatter(ex, ey, c="red", s=60, marker="s", label="Exit")
plt.xlabel("y")
plt.ylabel("x")
plt.title("Spatial congestion heatmap (Monte-Carlo averaged)")
plt.colorbar(shrink=0.8)
plt.tight_layout()

heatmap_path = os.path.join(out_dir, "spatial_congestion_heatmap.png")
plt.savefig(heatmap_path)
plt.close()

print(f"Saved spatial congestion heatmap to {heatmap_path}")


