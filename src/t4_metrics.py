import random
import numpy as np
from statistics import median
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns

from simulation import Simulation
from floorEnvironment import FloorEnvironment

# Configuration
FLOORS = ["t4_simple", "t4_big_door", "t4_pillar", "t4_funnel", "t4_rooms", "t4_chokepoints"]
AGENTS = [50, 100, 150]
K_VALS = [0.5, 1, 3, 5]
XI_VALS = [0.1, 0.5, 0.9]
METRICS_COUNT = 3


def run_simulation(params):
    """Wrapper function for parallel execution"""
    floor, agents, k, xi, f_idx, a_idx, k_idx, x_idx = params

    sff_path = f"data/floorPlansSSF/{floor}_sff.npy"
    floor_path = f"data/floorPlans/{floor}.fplan"

    steps, collisions, blocked = [], [], []

    for _ in range(5):
        seed = random.randint(0, 100_000_000)
        floor_env = FloorEnvironment(seed=seed, floor_layout_path=floor_path, floor_sff_path=sff_path)
        simulation = Simulation(np.random.default_rng(seed), floor_env.floor_layout, floor_env.floor_sff, agents, k, xi)

        while not simulation.is_completed():
            simulation.step()

        steps.append(simulation.metrics.steps_taken)
        collisions.append(simulation.metrics.collisions)
        blocked.append(simulation.metrics.blocked)

    result = [median(steps), median(collisions), median(blocked)]

    # Return result along with indices to place it correctly in the tensor
    return f_idx, a_idx, k_idx, x_idx, result


def analyze_impact(results):
    variables = {"Floor": (0, FLOORS), "Agents": (1, AGENTS), "K": (2, K_VALS), "Xi": (3, XI_VALS)}

    print("\n--- Interaction Analysis: K vs XI ---")
    # Average over Floor (axis 0) and Agents (axis 1)
    # This leaves us with a shape of (len(K_VALS), len(XI_VALS), METRICS_COUNT)
    interaction = np.mean(results, axis=(0, 1))

    metric_names = ["Steps", "Collisions", "Blocked"]
    for m_idx, name in enumerate(metric_names):
        print(f"\nMatrix for {name} (Rows=K, Cols=Xi):")
        # Header
        header = "      " + " ".join([f"Xi={x:<6}" for x in XI_VALS])
        print(header)

        for k_idx, k_val in enumerate(K_VALS):
            row = [f"{interaction[k_idx, x_idx, m_idx]:<9.1f}" for x_idx in range(len(XI_VALS))]
            print(f"K={k_val:<3} | " + " ".join(row))

    print("\n--- Variable Impact Analysis (Marginal Means) ---")
    for var_name, (axis, values) in variables.items():
        print(f"\nImpact of {var_name}:")
        axes_to_average = tuple(i for i in range(4) if i != axis)
        impact = np.mean(results, axis=axes_to_average)
        for i, val in enumerate(values):
            m = impact[i]
            print(f"  {val} -> Steps: {m[0]:.1f}, Collisions: {m[1]:.1f}, Blocked: {m[2]:.1f}")


def plot_interaction(results):
    interaction = np.mean(results, axis=(0, 1))
    metric_names = ["Steps", "Collisions", "Blocked"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, name in enumerate(metric_names):
        sns.heatmap(
            interaction[:, :, i],
            annot=True,
            fmt=".1f",
            xticklabels=XI_VALS,
            yticklabels=K_VALS,
            ax=axes[i],
            cmap="YlGnBu"
        )
        axes[i].set_title(f"K vs Xi Interaction ({name})")
        axes[i].set_xlabel("Xi")
        axes[i].set_ylabel("K")

    plt.tight_layout()
    plt.savefig("logs/interaction_heatmap.png")
    plt.show()


if __name__ == "__main__":
    results_tensor = np.zeros((len(FLOORS), len(AGENTS), len(K_VALS), len(XI_VALS), METRICS_COUNT))

    # 1. Prepare tasks
    tasks = []
    for f_idx, floor in enumerate(FLOORS):
        for a_idx, agents in enumerate(AGENTS):
            for k_idx, k in enumerate(K_VALS):
                for x_idx, xi in enumerate(XI_VALS):
                    tasks.append((floor, agents, k, xi, f_idx, a_idx, k_idx, x_idx))

    # 2. Execute in parallel
    print(f"Starting simulation with {len(tasks)} tasks...")
    # max_workers defaults to number of processors on the machine
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_simulation, tasks))

    # 3. Populate Tensor (Ordering is maintained by mapping indices)
    for f_idx, a_idx, k_idx, x_idx, res in results:
        results_tensor[f_idx, a_idx, k_idx, x_idx] = res
        print(
            f"Simulation(floor={FLOORS[f_idx]}, agents={AGENTS[a_idx]}, k={K_VALS[k_idx]}, xi={XI_VALS[x_idx]}) -> Metrics(steps={res[0]}, collisions={res[1]}, blocked={res[2]})",
            flush=True
        )

    analyze_impact(results_tensor)
    plot_interaction(results_tensor)
    np.save("logs/simulation_results.npy", results_tensor)
