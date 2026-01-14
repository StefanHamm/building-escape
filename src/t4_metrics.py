import random
import numpy as np
from statistics import median
from concurrent.futures import ProcessPoolExecutor
import loader as ld

from simulation import Simulation

# Configuration
FLOORS = ["t4_simple", "t4_big_door", "t4_double_exit", "t4_pillar", "t4_funnel", "t4_rooms"]
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

    for _ in range(10):
        simulation = Simulation(
            np.random.default_rng(),
            ld.loadFloorPlan(floor_path),
            ld.load_sff_from_npy(sff_path),
            [ld.load_sff_from_npy(sff_path)],
            agents,
            k,
            xi
        )

        while not simulation.is_completed():
            simulation.step()

        steps.append(simulation.metrics.steps_taken)
        collisions.append(simulation.metrics.collisions)
        blocked.append(simulation.metrics.blocked)

    result = [median(steps), median(collisions), median(blocked)]

    # Return result along with indices to place it correctly in the tensor
    return f_idx, a_idx, k_idx, x_idx, result


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

    np.save("logs/simulation_results.npy", results_tensor)
