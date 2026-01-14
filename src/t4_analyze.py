import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration (Ensure these match your simulation exactly)
FLOORS = ["t4_simple", "t4_big_door", "t4_double_exit", "t4_pillar", "t4_funnel", "t4_rooms"]
AGENTS = [50, 100, 150]
K_VALS = [0.5, 1, 3, 5]
XI_VALS = [0.1, 0.5, 0.9]
METRIC_IDX = 0  # 0: Steps, 1: Collisions, 2: Blocked


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


def plot_efficiency_grid(data):
    # Create the figure
    # Rows = Agent Counts, Cols = Floor Plans
    fig, axes = plt.subplots(len(AGENTS), len(FLOORS),
                             figsize=(16, 12),
                             sharex=True, sharey=True)

    # Find global min/max for Steps to ensure consistent color scaling
    v_min = np.min(data[..., METRIC_IDX])
    v_max = np.max(data[..., METRIC_IDX])

    for a_idx, agent_count in enumerate(AGENTS):
        for f_idx, floor_name in enumerate(FLOORS):
            ax = axes[a_idx, f_idx]

            # Extract the K vs Xi slice for this specific Floor and Agent count
            # Shape results in (len(K_VALS), len(XI_VALS))
            heatmap_data = data[f_idx, a_idx, :, :, METRIC_IDX]

            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt=".1f",
                cmap="YlGnBu",
                cbar=(f_idx == len(FLOORS) - 1),  # Only show colorbar on the last column
                ax=ax,
                xticklabels=XI_VALS,
                yticklabels=K_VALS,
                vmin=v_min,
                vmax=v_max
            )

            # Labels
            if a_idx == 0:
                ax.set_title(f"{floor_name}", fontweight='bold', pad=15)
            if f_idx == 0:
                ax.set_ylabel(f"{agent_count} Agents\n\nK (Intensity)", fontweight='bold')
            if a_idx == len(AGENTS) - 1:
                ax.set_xlabel("Xi (Friction)", fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("logs/floor_performance_grid.png")
    plt.show()


def plot_floor_vs_agents(data, k_val, xi_val):
    """
    Generates a heatmap comparing Floor Plans (X) vs Agent Density (Y)
    for a specific behavioral configuration.
    """
    k_idx = K_VALS.index(k_val)
    xi_idx = XI_VALS.index(xi_val)
    comparison_matrix = data[:, :, k_idx, xi_idx, 0].T

    # 3. Plotting
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        comparison_matrix,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        xticklabels=FLOORS,
        yticklabels=AGENTS,
    )

    plt.xlabel("Floor Plan Structure", fontweight='bold')
    plt.ylabel("Number of Agents (Density)", fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"logs/scaling_comparison_k{k_val}_xi{xi_val}.png")
    plt.show()


def print_verdict(data):
    print("--- Global Performance Verdict (Average Steps) ---")
    for f_idx, floor in enumerate(FLOORS):
        avg_steps = np.mean(data[f_idx, ..., METRIC_IDX])
        print(f"{floor:<20}: {avg_steps:.2f} avg steps")


if __name__ == "__main__":
    path = "logs/simulation_results.npy"
    data = np.load(path)

    analyze_impact(data)
    plot_interaction(data)

    plot_efficiency_grid(data)
    plot_floor_vs_agents(data, k_val=5, xi_val=0.5)
