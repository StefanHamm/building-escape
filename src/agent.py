# -Chooses actions based on observations observations are passed from the environment
# -Implements movement strategy
# -Tracks personal state
# agent interacts with environment and gets a new state position
# full moore neighborhood


from sharedClasses import Observation, AgentState, Actions
import numpy as np


class Agent:

    def __init__(self, id, start_state, k, rng: np.random.Generator, decisionType: str, all_goals_sff: np.ndarray,
                 personalized_sff: np.ndarray,disable_personalized_exit=False,disable_agent_greedy_k=False):
        self.id = id
        self.state = start_state
        self.rng = rng
        # make agents randomly more greedy
        if disable_agent_greedy_k:
            self.k = k
        else:
            self.k = max(0.3, self.rng.normal(loc=k, scale=k / 3))
        if self.k <= 0:
            raise ValueError("Parameter k must be positive")
        self.memory = [self.state]  # to store past states or observations if needed
        self.actions = Actions()
        self.decisionType = decisionType  # e.g., default, min_scaling, division_scaling
        self.mobility = np.clip(self.rng.normal(0.8, 0.2), 0.3, 1.0)
        self.all_goals_sff = all_goals_sff
        if disable_personalized_exit:
            self.personalized_sff = all_goals_sff
        else:
            self.personalized_sff = personalized_sff

    def _get_moore_neighborhood(self, size=3):
        # Create a window initialized with Infinity (Walls)
        window = np.full((size, size), np.inf)

        # Determine bounds
        x_start, x_end = max(0, self.state.x - 1), min(self.all_goals_sff.shape[0], self.state.x + 2)
        y_start, y_end = max(0, self.state.y - 1), min(self.all_goals_sff.shape[1], self.state.y + 2)

        # Determine placement in window
        win_x_start = 1 - (self.state.x - x_start)
        win_x_end = win_x_start + (x_end - x_start)
        win_y_start = 1 - (self.state.y - y_start)
        win_y_end = win_y_start + (y_end - y_start)

        if self.all_goals_sff[self.state.x, self.state.y] <= 5:
            # Agents close to exit can see it, so they use it
            sff = self.all_goals_sff
        else:
            # Agents not close to exit choose one they know no matter how far
            sff = self.personalized_sff
        window[win_x_start:win_x_end, win_y_start:win_y_end] = sff[x_start:x_end, y_start:y_end]

        return window

    def decide_action(self):

        movement = Observation(self._get_moore_neighborhood())

        movement.mooreNeighborhoodSFF[1, 1] = np.inf
        # swich case for self.decisionType
        if self.decisionType == "default":
            pass  # no modification
        elif self.decisionType == "min_scaling":
            # set 1,1 in moore neighborhood to inf to ignore it in min calculation
            min_sff = np.min(movement.mooreNeighborhoodSFF)
            movement.mooreNeighborhoodSFF -= min_sff
        elif self.decisionType == "division_scaling":
            min_sff = np.min(movement.mooreNeighborhoodSFF)
            if min_sff != 0:
                movement.mooreNeighborhoodSFF /= min_sff

        probability_matrix = np.exp(-self.k * movement.mooreNeighborhoodSFF)
        flattened_probs = probability_matrix.flatten()
        total_prob = flattened_probs.sum()  # normalize to sum to 1

        # if surrounded by walls stay put/stuck
        if total_prob == 0:
            return (0, 0)  # No valid move available

        flattened_probs /= total_prob

        chosen_index = self.rng.choice(len(flattened_probs), p=flattened_probs)
        move_direction = self.actions.MOORE_ACTIONS[chosen_index]

        if move_direction is None:
            return (0, 0)
        return move_direction  # adjust for offset

    def update_state(self, new_state: AgentState):
        self.memory.append(new_state)
        self.state = new_state
        if new_state.done:
            print(f"Agent {self.id} has reached the exit.")
            return 1
        return 0


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    agent = Agent(id=1, start_state=AgentState(5, 5), k=1, rng=rng)
    obs = Observation(np.array([[0, np.inf, np.inf],
                                [np.inf, 0.0, np.inf],
                                [np.inf, np.inf, 1]]))

    action = agent.decide_action(obs)
    print(f"Agent decided to move: {action}")
