import dataclasses
import random

import numpy as np
from helper import getSafeWhiteCoords
from sharedClasses import AgentState
from agent import Agent


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
    def __init__(self, rng: np.random.Generator, floor_layout, all_goals_sff: np.ndarray,
                 goal_specific_sffs: list[np.ndarray], agent_count, k, xi):
        assert floor_layout.shape[0] == all_goals_sff.shape[0]
        assert floor_layout.shape[1] == all_goals_sff.shape[1]

        self.rng = rng
        self.floor_layout = floor_layout
        self.all_goals_sff = all_goals_sff
        self.goal_specific_sffs = goal_specific_sffs
        self.agent_count = agent_count
        self.k = k
        self.xi = xi
        self.x_dim = self.floor_layout.shape[0]
        self.y_dim = self.floor_layout.shape[1]
        self.agentmap = SpatialPool()
        self.metrics = Metrics()

        # 1. Get all traversable coordinates
        free_space = list(getSafeWhiteCoords(self.floor_layout, self.all_goals_sff))
        actual_target = min(len(free_space), self.agent_count)

        # 2. Cluster Spawning Logic
        selected_coords = []
        available_pool = free_space.copy()

        while len(selected_coords) < actual_target:
            # Pick a random "seed" from the remaining available space
            seed_idx = self.rng.choice(len(available_pool))
            seed_coord = available_pool.pop(seed_idx)
            selected_coords.append(seed_coord)

            # Determine how many more agents to add to this specific cluster
            remaining_needed = actual_target - len(selected_coords)
            current_cluster_target = min(random.randint(1, 20), remaining_needed, len(available_pool))

            if current_cluster_target > 0:
                # Calculate Euclidean distance from all remaining points to the seed
                coords_array = np.array(available_pool)
                distances = np.linalg.norm(coords_array - np.array(seed_coord), axis=1)

                # Get indices of the closest points
                closest_indices = np.argsort(distances)[:current_cluster_target]

                # Pull them out of the available pool and add to our selection
                # We sort indices in reverse to pop correctly without shifting index references
                for idx in sorted(closest_indices, reverse=True):
                    selected_coords.append(available_pool.pop(idx))

        # 3. Place the agents
        for i, (x, y) in enumerate(selected_coords):
            # Agent knows random subset of exits
            num_exits = len(self.goal_specific_sffs)
            num_to_know = self.rng.integers(1, num_exits + 1)
            known_indices = self.rng.choice(num_exits, size=num_to_know, replace=False)

            # Combine the SFFs: The agent follows the shortest path to ANY known exit
            # Initialize with infinity
            agent_sff = np.full((self.x_dim, self.y_dim), np.inf)
            for idx in known_indices:
                agent_sff = np.minimum(agent_sff, self.goal_specific_sffs[idx])
            agent = Agent(i + 1, AgentState(x, y), self.k, self.rng, "default", all_goals_sff, agent_sff)
            self.agentmap.add(agent, x, y)

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
            if self.rng.random() >= agent.mobility:
                # Slow agents move less
                continue
            dy, dx = agent.decide_action()

            target_x = agent.state.x + dx
            target_y = agent.state.y + dy

            if (target_x, target_y) not in proposals:
                proposals[(target_x, target_y)] = []
            proposals[(target_x, target_y)].append(agent)

        # Phase 2: resolve collisions
        agents_to_move: dict[int, tuple[Agent, int, int]] = {}
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
                if self.all_goals_sff[tx, ty] == 0:
                    agents_to_remove.append((winner, tx, ty))
                # Move only if the target is empty
                else:
                    agents_to_move[winner.id] = (winner, tx, ty)

        # Phase 3: Execute
        for agent, tx, ty in agents_to_remove:
            self.agentmap.remove(agent)
        while True:
            to_remove = []
            for aid, (agent, tx, ty) in agents_to_move.items():
                if self.agentmap.get_at(tx, ty) is None:
                    self.agentmap.unsafe_update_grid(agent, tx, ty)
                    to_remove.append(aid)
            if len(to_remove) == 0:
                self.metrics.blocked += len(agents_to_move)
                break
            for aid in to_remove:
                del agents_to_move[aid]
        self.metrics.steps_taken += 1

    def is_completed(self):
        return all(map(lambda x: x.state.done, self.agentmap.agents))
