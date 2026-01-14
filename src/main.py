import glob

from simulation import Simulation
import numpy as np
from visualize import create_video_from_steps, print_agents_on_floorplan, floorplan_to_rgb
import shutil
import time
import os
import multiprocessing
import copy
import timeit
import loader as ld


def render_console(simulation, step_num=0):
    """
    Prints the grid state to console.
    
    Handles:
    - Layout: numpy array of characters (dtype='U1').
    - SFF: numpy array of floats (0.0 is Exit).
    """
    # --- Configuration ---
    # Define what characters in your .fplan file count as walls
    WALL_CHARS = {'#', '1', 'W', '@'}

    # ANSI Colors for terminal output
    RED = '\033[91m'  # Agents
    GREEN = '\033[92m'  # Exits
    WHITE = '\033[97m'  # Walls
    RESET = '\033[0m'

    # Visual Blocks
    WALL_BLOCK = f"{WHITE}██{RESET}"
    AGENT_BLOCK = f"{RED}(){RESET}"
    EXIT_BLOCK = f"{GREEN}XX{RESET}"
    EMPTY_BLOCK = "  "

    output_buffer = []

    # Header
    active_count = len(simulation.agentmap)
    finished_count = len(simulation.agentmap.finished_agents)
    output_buffer.append(f"--- Step: {step_num:03d} | Active: {active_count} | Exited: {finished_count} ---")

    rows, cols = simulation.x_dim, simulation.y_dim

    for x in range(rows):
        row_str = ""
        for y in range(cols):

            agent = simulation.agentmap.get_at(x, y)
            if agent is not None:
                row_str += AGENT_BLOCK


            elif simulation.all_goals_sff[x, y] == 0:
                row_str += EXIT_BLOCK

            elif simulation.floor_layout[x, y] in WALL_CHARS:
                row_str += WALL_BLOCK

            else:
                row_str += EMPTY_BLOCK

        output_buffer.append(row_str)

    full_frame = "\n".join(output_buffer)

    # os.system('cls' if os.name == 'nt' else 'clear')
    print(full_frame)


def save_frame_async(floor_layout, agents, step, export_path, base_rgb_img=None):
    """Worker function to save frame."""
    print_agents_on_floorplan(floor_layout, agents, step=step, export_path=export_path, base_rgb=base_rgb_img)


RENDER = True
AGENTS = 700
FLOOR = "freihausEG"

if __name__ == "__main__":
    total_start = timeit.default_timer()

    sffs = [ld.load_sff_from_npy(x) for x in glob.glob(f"data/floorPlansSSF/{FLOOR}_sff_*.npy")]
    floor_path = f"data/floorPlans/{FLOOR}.fplan"

    rng = np.random.default_rng()
    simulation = Simulation(
        rng,
        ld.loadFloorPlan(floor_path),
        ld.load_sff_from_npy(f"data/floorPlansSSF/{FLOOR}_sff.npy"),
        sffs,
        AGENTS,
        5,
        0.5,
        True,  # disable_personalized_exit
        True,  # disable_cluster_spawn
        True,  # disable_agent_mobility
        True   # disable_agent_greedy_k
    )

    # Precompute RGB floorplan once
    base_rgb_img = None
    if RENDER:
        base_rgb_img = floorplan_to_rgb(simulation.floor_layout)

    # render_console(simulation, 0)
    time.sleep(1.0)

    pool = None
    async_results = []
    if RENDER:
        # clear the original logs/steps folder BEFORE creating the pool
        log_dir = f"logs/steps/{FLOOR}/"
        os.makedirs(log_dir, exist_ok=True)
        for filename in os.listdir(log_dir):
            file_path = os.path.join(log_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Warning: Failed to delete {file_path}. Reason: {e}")

        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        export_path = f"logs/steps/{FLOOR}/0.png"
        # Pass copy of agents since objects are modified in main process
        agents_copy = copy.deepcopy(simulation.agentmap.agents)
        res = pool.apply_async(save_frame_async, (simulation.floor_layout, agents_copy, 0, export_path, base_rgb_img))
        async_results.append(res)

    for step in range(400):
        simulation.step()

        #render_console(simulation, step)

        # Check completion
        if simulation.is_completed():
            print(simulation.metrics)
            break

        if RENDER:
            # Clean up finished tasks to manage RAM
            async_results = [res for res in async_results if not res.ready()]

            # Throttle main loop if rendering falls behind (limit to 2x CPU count)
            # This prevents unlimited RAM usage from queued up agent copies
            while len(async_results) >= multiprocessing.cpu_count() * 2:
                time.sleep(0.05)
                async_results = [res for res in async_results if not res.ready()]

            export_path = f"logs/steps/{FLOOR}/{step + 1}.png"
            # Pass copy of agents since objects are modified in main process
            agents_copy = copy.deepcopy(simulation.agentmap.agents)
            res = pool.apply_async(save_frame_async,
                                   (simulation.floor_layout, agents_copy, step + 1, export_path, base_rgb_img))
            async_results.append(res)

    if RENDER:
        # render last frame
        export_path = f"logs/steps/{FLOOR}/{step + 2}.png"
        agents_copy = copy.deepcopy(simulation.agentmap.agents)
        res = pool.apply_async(save_frame_async,
                               (simulation.floor_layout, agents_copy, step + 2, export_path, base_rgb_img))
        async_results.append(res)

    if RENDER and pool:
        print("Waiting for image generation to finish...")
        for res in async_results:
            res.get()
        pool.close()
        pool.join()

        os.makedirs("logs/video/", exist_ok=True)
        create_video_from_steps(f"logs/steps/{FLOOR}/", f"logs/video/{FLOOR}.mp4", fps=5)

    total_end = timeit.default_timer()
    # print(f"Total simulation and rendering time: {total_end - total_start:.2f} seconds")
