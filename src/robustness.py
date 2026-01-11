import numpy as np
from simulation import Simulation

def run_monte_carlo(
    floor_layout,
    layout_sff,
    agent_count,
    k,
    xi,
    n_runs=100,
    max_steps=10_000
):
    evacuation_times = []

    for r in range(n_runs):
        rng = np.random.default_rng(seed=r)

        sim = Simulation(
            rng=rng,
            floor_layout=floor_layout,
            goal_specific_sffs=layout_sff,
            agent_count=agent_count,
            k=k,
            xi=xi,
            verbose=0
        )

        while not sim.is_completed() and sim.step_count < max_steps:
            sim.step()

        evacuation_times.append(sim.step_count)

    evacuation_times = np.array(evacuation_times)

    return {
        "mean_time": evacuation_times.mean(),
        "std_time": evacuation_times.std(),
        "max_time": evacuation_times.max(),
        "min_time": evacuation_times.min(),
        "all_times": evacuation_times
    }

def run_spatial_heatmap(
    floor_layout,
    layout_sff,
    agent_count,
    k,
    xi,
    n_runs=50,
    max_steps=5000
):
    heat = np.zeros_like(floor_layout, dtype=float)

    for r in range(n_runs):
        rng = np.random.default_rng(seed=r)

        sim = Simulation(
            rng=rng,
            floor_layout=floor_layout,
            goal_specific_sffs=layout_sff,
            agent_count=agent_count,
            k=k,
            xi=xi,
            verbose=0
        )

        while not sim.is_completed() and sim.step_count < max_steps:
            # count agent positions BEFORE movement
            for agent in sim.agentmap.agents:
                if not agent.state.done:
                    heat[agent.state.x, agent.state.y] += 1

            sim.step()

    # normalize to [0,1] for visualization
    if heat.max() > 0:
        heat /= heat.max()

    return heat
