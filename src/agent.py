


#-Chooses actions based on observations observations are passed from the environment
#-Implements movement strategy
#-Tracks personal state
# agent interacts with environment and gets a new state position
#full moore neighborhood


from sharedClasses import Observation, AgentState,AgentAction,Actions
import numpy as np

class Agent:

    def __init__(self,agent_id, start_state,k,seed=42):
        self.agent_id = agent_id
        self.state = start_state
        self.rng = np.random.default_rng(seed)
        self.k = k  # model parameter must be positive
        if self.k <= 0:
            raise ValueError("Parameter k must be positive")
        self.memory = [self.state]  # to store past states or observations if needed
        self.actions = Actions()
        
   
    
    def decide_action(self, observation:Observation):
        probability_matrix = np.exp(-self.k * observation.mooreNeigbhborhoodSFF) 
        # set 11 to zero since it's the agent's current position
        print(probability_matrix)
        probability_matrix[1, 1] = 0
        # draw action based on the probability distribution
        flattened_probs = probability_matrix.flatten()
        flattened_probs /= flattened_probs.sum()  # normalize to sum to 1
        chosen_index = self.rng.choice(len(flattened_probs), p=flattened_probs)
        move_direction = self.actions.MOORE_ACTIONS[chosen_index]
        print(chosen_index)
        return move_direction  # adjust for offset
        
    def update_state(self, new_state:AgentState):
        self.memory.append(new_state)
        self.state = new_state
        if new_state.done:
            print(f"Agent {self.agent_id} has reached the exit.")
            return 1
        return 0
        
        
    
if __name__ == "__main__":
    agent = Agent(agent_id=1, start_state=AgentState(5, 5), k=1, seed=42)
    obs = Observation(np.array([[0, np.inf, np.inf],
                                [np.inf, 0.0, np.inf],
                                [np.inf, np.inf, 1]]))
    
    action = agent.decide_action(obs)
    print(f"Agent decided to move: {action}")