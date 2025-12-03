
import loader as ld


#Floor plan, SFF, occupancy map
#Physics/collision checking
#Timestep advancement
# Terminal state = reached exit

class FloorEnvironment:
    def __init__(self, seed, floor_layout_path, floor_sff_path, agent_count=10, k = 1):
        self.seed = seed
        self.floor_layout_path = floor_layout_path
        self.floor_sff_path = floor_sff_path
        self.floor_layout = ld.loadFloorPlan(self.floor_layout_path)
        self.floor_sff = ld.load_sff_from_npy(self.floor_sff_path)



            


        
        
        
        
        
        