import dataclasses

import numpy as np
from helper import getSafeWhiteCoords
from sharedClasses import AgentState, Observation
from agent import Agent

from visualize import print_agents_on_floorplan


@dataclasses.dataclass
class Metrics:
    steps_taken: int
    collisions: int
    blocked: int

    def __init__(self, steps_taken=0, collisions=0, blocked=0):
        self.steps_taken = steps_taken
        self.collisions = collisions
        self.blocked = blocked


class SpatialPool:
    def __init__(self):
        self.agents = []  # Dense array for iteration
        self.finished_agents = []  # Archive for post-processing
        self.id_to_idx = {}  # ID -> Index (fast deletion)
        self.grid: dict[tuple[int, int], Agent] = {}  # (x, y) -> Agent for spacial lookup

    def __iter__(self):
        return iter(self.agents)

    def __len__(self):
        return len(self.agents)

    def add(self, agent: Agent, x, y):
        if (x, y) in self.grid:
            raise ValueError(f"Position {x},{y} already occupied.")

        self.grid[(x, y)] = agent
        self.agents.append(agent)
        self.id_to_idx[agent.id] = len(self.agents) - 1

    def remove(self, agent: Agent):
        if agent.id not in self.id_to_idx: return

        agent.state.done = True
        self.finished_agents.append(agent)  # TODO handle update_state in agents

        if self.grid.get((agent.state.x, agent.state.y)).id == agent.id:
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
        old_pos = (agent.state.x, agent.state.y)
        agent_at_pos = self.grid.get(old_pos)
        if agent_at_pos and agent_at_pos.id == agent.id:
            del self.grid[old_pos]
        self.grid[(new_x, new_y)] = agent
        agent.state.x = new_x
        agent.state.y = new_y


class Simulation:
    def __init__(self, rng: np.random.Generator, floor_layout, layout_sff, agent_count, k, xi, verbose=0):

        self.verbose = verbose  # 1: print basic info, 2: detailed per-step info 0: silent
        self.rng = rng
        self.floor_layout = floor_layout
        self.layout_sff = layout_sff
        self.agent_count = agent_count
        self.k = k
        self.xi = xi
        self.x_dim = self.floor_layout.shape[0]
        self.y_dim = self.floor_layout.shape[1]
        self.agentmap = SpatialPool()
        self.metrics = Metrics()

        free_space = list(getSafeWhiteCoords(self.floor_layout, self.layout_sff))
        actual_count = min(len(free_space),
                           self.agent_count)  # Ensure we don't try to spawn more agents than free space
        selected_idx = self.rng.choice(len(free_space), size=actual_count, replace=False)

        for i, idx in enumerate(selected_idx):
            (x, y) = free_space[idx]
            agent = Agent(i + 1, AgentState(x, y), self.k, self.rng, self.verbose)
            self.agentmap.add(agent, x, y)

        if len(self.agentmap) < self.agent_count:
            if self.verbose >= 1:
                print(f"Warning: Only {len(self.agentmap)} agents were placed due to limited free space.")

    def _get_moore_neighborhood(self, x, y):
        # Create a 3x3 window initialized with Infinity (Walls)
        window = np.full((3, 3), np.inf)

        # Determine bounds
        x_start, x_end = max(0, x - 1), min(self.x_dim, x + 2)
        y_start, y_end = max(0, y - 1), min(self.y_dim, y + 2)

        # Determine placement in window
        win_x_start = 1 - (x - x_start)
        win_x_end = win_x_start + (x_end - x_start)
        win_y_start = 1 - (y - y_start)
        win_y_end = win_y_start + (y_end - y_start)

        window[win_x_start:win_x_end, win_y_start:win_y_end] = \
            self.layout_sff[x_start:x_end, y_start:y_end]

        return window

    def _calculate_friction_probability(self, n):
        assert n > 1
        # Formula: 1 - (1 - xi)^n - n*xi*(1 - xi)^(n-1)
        term1 = (1 - self.xi) ** n
        term2 = n * self.xi * ((1 - self.xi) ** (n - 1))
        return 1.0 - term1 - term2

    def step(self):
        # Phase 1: plan move
        proposals: dict[tuple[int, int], list[Agent]] = {}
        current_agents = [a for a in self.agentmap.agents if not a.state.done]
        self.rng.shuffle(current_agents)

        for agent in current_agents:
            if self.verbose >= 1:
                print("TSTE")
            obs = Observation(self._get_moore_neighborhood(agent.state.x, agent.state.y))
            dy, dx = agent.decide_action(obs)
            if self.verbose >= 1:
                print(agent.verbose)

            target_x = agent.state.x + dx
            target_y = agent.state.y + dy

            if (target_x, target_y) not in proposals:
                proposals[(target_x, target_y)] = []
            proposals[(target_x, target_y)].append(agent)

        # Phase 2: resolve collisions
        agents_to_move: list[tuple[Agent, int, int]] = []
        agents_to_remove: list[tuple[Agent, int, int]] = []

        for target, candidates in proposals.items():
            n = len(candidates)
            winner = None
            if n == 1:
                winner = candidates[0]
            else:
                mu = self._calculate_friction_probability(n)
                if self.rng.random() >= mu:
                    winner = self.rng.choice(candidates)
                else:
                    self.metrics.collisions += 1
            if winner:
                tx, ty = target
                assert 0 <= tx < self.x_dim and 0 <= ty < self.y_dim, "Agent wants to move outside map!"
                # Check for exit (SFF == 0)
                if self.layout_sff[tx, ty] == 0:
                    agents_to_remove.append((winner, tx, ty))
                # Move only if the target is empty
                elif self.agentmap.get_at(tx, ty) is None:
                    agents_to_move.append((winner, tx, ty))
                else:
                    self.metrics.blocked += 1

        # Phase 3: Execute
        for agent, tx, ty in agents_to_remove:
            self.agentmap.remove(agent)
        for agent, tx, ty in agents_to_move:
            self.agentmap.unsafe_update_grid(agent, tx, ty)

        if self.verbose >= 2:
            # file path is logs/steps/step_{self.metrics.steps_taken}.png
            filePath = f"logs/steps/{self.metrics.steps_taken}.png"
            print_agents_on_floorplan(self.floor_layout, self.agentmap.agents, filePath)
        self.metrics.steps_taken += 1

    def is_completed(self):
        return all(map(lambda x: x.state.done, self.agentmap.agents))
