import random
from sharedClasses import AgentState, Actions, AgentAction
from agent import Agent

class Simulation:
    def __init__(self, seed, floor_layout_path, floor_sff_path, agent_count, k):
        self.agent_count = agent_count
        self.k = k
        random.seed(self.seed)
        self.x_dim = self.floor_layout.shape[0]
        self.y_dim = self.floor_layout.shape[1]
        self.agents = [] #TBD find appropriate data structure for fast iteration, fast insert and fast delete -> gemini: swap&pop or slot map
        self.spatial_map = {} # Spatial Map {(x,y) -> Agent}
        for i in range(1,self.agent_count):
            self.add_agent(i)

    def is_field_occupied(self, x, y):
        return (x,y) in self.spatial_map
    
    # currently random assignment, change me for
    def add_agent(self, id):
        while True:
            x = random.randint(0, self.x_dim)
            y = random.randint(0, self.y_dim)
            coord = (x,y)

            if self.is_field_occupied(x,y):
                continue
            
            agent = Agent(id, AgentState(x,y, self.k))
            self.agents.append(agent)
            self.spatial_map[coord] = agent

    def run(self):
        pass
            
            
if __name__ == "__main__":
    sim = Simulation()
    sim.run()
