#from floorEnvironment import FloorEnvironment
from simulation import Simulation
from agent import Agent
from floorEnvironment import FloorEnvironment
import numpy as np
from visualize import create_video_from_steps

if __name__ == "__main__":
    sff_path = "data/floorPlansSSF/small_sff.npy"
    floor_path = "data/floorPlans/small.fplan"

    floor_env = FloorEnvironment(seed=42, floor_layout_path=floor_path, floor_sff_path=sff_path, agent_count=5, k=0.1)
    
    rng = np.random.default_rng(61)
    
    Simulation_instance = Simulation(rng,floor_env.floor_layout,floor_env.floor_sff,15,5,5,2)
    
    
    for step in range(10):
        Simulation_instance.step()
    
    create_video_from_steps("logs/steps/", "logs/video/simulation_output.mp4", fps=1)
    
    
    
    