#from floorEnvironment import FloorEnvironment
from simulation import Simulation
from agent import Agent
from floorEnvironment import FloorEnvironment
import numpy as np
#from visualize import create_video_from_steps,print_agents_on_floorplan

import time
import sys
import os

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
    RED = '\033[91m'     # Agents
    GREEN = '\033[92m'   # Exits
    WHITE = '\033[97m'   # Walls
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

            
            elif simulation.layout_sff[x, y] == 0:
                row_str += EXIT_BLOCK
                
            elif simulation.floor_layout[x, y] in WALL_CHARS: 
                row_str += WALL_BLOCK
                
            else:
                row_str += EMPTY_BLOCK
        
        output_buffer.append(row_str)

    full_frame = "\n".join(output_buffer)
    
    #os.system('cls' if os.name == 'nt' else 'clear')
    print(full_frame)
    
RENDER = True

if __name__ == "__main__":
    #sff_path = "data/floorPlansSSF/small_sff.npy"
    #floor_path = "data/floorPlans/small.fplan"
    
    floor = "small"
    sff_path = f"data/floorPlansSSF/{floor}_sff.npy"
    floor_path = f"data/floorPlans/{floor}.fplan"
    floor_env = FloorEnvironment(seed=42, floor_layout_path=floor_path, floor_sff_path=sff_path, agent_count=5, k=0.1)
    
    rng = np.random.default_rng()

    Simulation_instance = Simulation(rng,floor_env.floor_layout,floor_env.floor_sff,10,5,10,0)
    render_console(Simulation_instance, 0)
    time.sleep(1.0)
    
    for step in range(400):
        Simulation_instance.step()

        render_console(Simulation_instance, step)
        time.sleep(0.2)
        
        # Check completion
        if Simulation_instance.is_completed():
            print("\nALL AGENTS EVACUATED!")
            break
