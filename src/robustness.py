import numpy as np
from simulation import Simulation

def run_monte_carlo(
    floor_layout,
    layout_sff,
    agent_count,
    k,
    xi,
    n_runs=100,
    max_steps=5000
):
    evacuation_times = []
    failed_runs = 0

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

        while not sim.is_completed() and sim.metrics.steps_taken < max_steps:
            sim.step()

        if sim.is_completed():
            # SUCCESS: true evacuation time
            evacuation_times.append(sim.metrics.steps_taken)
        else:
            # FAILURE: not fully evacuated
            failed_runs += 1

    evacuation_times = np.array(evacuation_times)

    return {
        "mean_time": evacuation_times.mean() if len(evacuation_times) > 0 else np.nan,
        "std_time": evacuation_times.std() if len(evacuation_times) > 0 else np.nan,
        "min_time": evacuation_times.min() if len(evacuation_times) > 0 else np.nan,
        "max_time": evacuation_times.max() if len(evacuation_times) > 0 else np.nan,
        "all_times": evacuation_times,
        "failure_rate": failed_runs / n_runs
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

        while not sim.is_completed() and sim.metrics.steps_taken < max_steps:
            # count agent positions BEFORE movement
            for agent in sim.agentmap.agents:
                if not agent.state.done:
                    heat[agent.state.x, agent.state.y] += 1

            sim.step()

    # normalize to [0,1] for visualization
    if heat.max() > 0:
        heat /= heat.max()

    return heat
