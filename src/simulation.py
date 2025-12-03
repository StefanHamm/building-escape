import numpy as np
from helper import getAllWhiteCoords
from sharedClasses import AgentState, Observation
from agent import Agent


class SpatialPool:
    def __init__(self):
        self.agents = [] # Dense array for iteration
        self.finished_agents = [] # Archive for post-processing
        self.id_to_idx = {} # ID -> Index (fast deletion)
        self.grid = {} # (x, y) -> Agent for spacial lookup
    
    def __iter__(self):
        return iter(self.agents)

    def __len__(self):
        return len(self.agents)
    
    def add(self, agent, x, y):
        if (x, y) in self.grid:
            raise ValueError(f"Position {x},{y} already occupied.")
        
        self.grid[(x,y)] = agent
        self.agents.append(agent)
        self.id_to_idx[agent.id] = len(self.agents) - 1
    
    def remove(self, agent: Agent):
        if agent.id not in self.id_to_idx: return

        agent.state.done = True
        self.finished_agents.append(agent) # TODO handle update_state in agents

        if self.grid.get((agent.state.x, agent.state.y)) == agent:
            del self.grid[(agent.state.x, agent.state.y)]
        
        idx = self.id_to_idx[agent.id]
        last_agent = self.agents[-1]
        self.agents[idx] = last_agent
        self.id_to_idx[last_agent.id] = idx

        self.agents.pop()
        del self.id_to_idx[agent.id]
        return agent
    
    def get_at(self, x, y):
        return self.grid.get((x, y))
    
    def unsafe_update_grid(self, agent, new_x, new_y):
        """
        Updates coordinate without checking collision. 
        Used only during the batch update phase where logic is pre-calculated.
        """
        del self.grid[(agent.state.x, agent.state.y)]
        self.grid[(new_x, new_y)] = agent
        agent.state.x = new_x
        agent.state.y = new_y

class Simulation:
    def __init__(self, rng: np.random.Generator, layout, layout_sff, agent_count, k, xi):
        self.rng = rng
        self.layout = layout
        self.layout_sff = layout_sff
        self.agent_count = agent_count
        self.k = k
        self.xi = xi
        self.x_dim = self.floor_layout.shape[0]
        self.y_dim = self.floor_layout.shape[1]
        self.agentmap = SpatialPool()

        free_space = list(getAllWhiteCoords(self.layout))
        actual_count = min(len(free_space), self.agent_count) # Ensure we don't try to spawn more agents than free space
        selected_idx = self.rng.choice(len(free_space), size=actual_count, replace=False)

        for i, idx in enumerate(selected_idx):
            (x,y) = free_space[idx]
            agent = Agent(i+1,AgentState(x,y),self.k, self.rng)
            self.agentmap.add(agent, x, y)

    def _get_moore_neighborhood(self, x, y):
        # Create a 3x3 window initialized with Infinity (Walls)
        window = np.full((3, 3), np.inf)
        
        # Determine bounds
        x_start, x_end = max(0, x-1), min(self.x_dim, x+2)
        y_start, y_end = max(0, y-1), min(self.y_dim, y+2)
        
        # Determine placement in window
        win_x_start = 1 - (x - x_start)
        win_x_end = win_x_start + (x_end - x_start)
        win_y_start = 1 - (y - y_start)
        win_y_end = win_y_start + (y_end - y_start)
        
        window[win_x_start:win_x_end, win_y_start:win_y_end] = \
            self.layout_sff[x_start:x_end, y_start:y_end]
            
        return window
    
    def _calculate_friction_probability(self, n):
        if n <= 1: return 0.0
        # Formula: 1 - (1 - xi)^n - n*xi*(1 - xi)^(n-1)
        term1 = (1 - self.xi)**n
        term2 = n * self.xi * ((1 - self.xi)**(n - 1))
        return 1.0 - term1 - term2

    def step(self):
        # Phase 1: plan move
        proposals = {}
        current_agents = list(self.agentmap.agents)
        self.rng.shuffle(current_agents)

        for agent in current_agents:
            obs = Observation(self._get_moore_neighborhood(agent.state.x, agent.state.y))
            dx, dy = agent.decide_action(obs)
            
            target_x = agent.state.x + dx
            target_y = agent.state.y + dy
            
            if (target_x, target_y) not in proposals:
                proposals[(target_x, target_y)] = []
            proposals[(target_x, target_y)].append(agent)

        # Phase 2: resolve collisions
        agents_to_move = []   
        agents_to_remove = []

        for target, candidates in proposals.items():
            n = len(candidates)
            winner = None
            if n == 1:
                winner = candidates[0]
            else:
                mu = self._calculate_friction_probability(n)
                if self.rng.random() >= mu:
                    winner = self.rng.choice(candidates)
            if winner:
                tx, ty = target
                # Check for exit (SFF == 0)
                if 0 <= tx < self.x_dim and 0 <= ty < self.y_dim:
                    if self.layout_sff[tx, ty] == 0:
                        agents_to_remove.append(winner)
                    else:
                        agents_to_move.append((winner, tx, ty))

        # Phase 3: Execute
        for agent in agents_to_remove:
            self.agentmap.remove(agent)
        
        final_moves = []
        moving_agent_ids = {a.id for a in agents_to_remove}
        for (a, _, _) in agents_to_move:
            moving_agent_ids.add(a.id)
        
        for agent, tx, ty in agents_to_move:
            occupier = self.agentmap.get_at(tx, ty)
            
            # Move only if the target is empty OR the current occupier is also moving away
            if occupier is None or occupier.id in moving_agent_ids:
                final_moves.append((agent, tx, ty))

        for agent, tx, ty in final_moves:
            self.agentmap.unsafe_update_grid(agent, tx, ty)
            # TODO handle update_state in agents

    def is_completed(self):
        return all(map(lambda x: x.state.done, self.agentmap.agents))
