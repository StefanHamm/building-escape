


#-Chooses actions based on observations observations are passed from the environment
#-Implements movement strategy
#-Tracks personal state
# agent interacts with environment and gets a new state position
#full moore neighborhood


from sharedClasses import Observation, AgentState,Actions
import numpy as np

class Agent:

    def __init__(self, id, start_state, k, rng: np.random.Generator,decisionType:str,vebose=0):
        self.id = id
        self.state = start_state
        self.verbose = vebose
        self.rng = rng
        self.k = k  # model parameter must be positive
        if self.k <= 0:
            raise ValueError("Parameter k must be positive")
        self.memory = [self.state]  # to store past states or observations if needed
        self.actions = Actions()
        self.decisionType = decisionType  # e.g., default, min_scaling, division_scaling
        
        
   
    
    def decide_action(self, observation:Observation):
        observation.mooreNeighborhoodSFF[1,1] = np.inf
        #swich case for self.decisionType
        if self.decisionType == "default":
            pass  # no modification
        elif self.decisionType == "min_scaling":
            #set 1,1 in moore neighborhood to inf to ignore it in min calculation
            min_sff = np.min(observation.mooreNeighborhoodSFF)
            observation.mooreNeighborhoodSFF -= min_sff
        elif self.decisionType == "division_scaling":
            min_sff = np.min(observation.mooreNeighborhoodSFF)
            if min_sff !=0:
                observation.mooreNeighborhoodSFF /= min_sff
            
        
        probability_matrix = np.exp(-self.k * observation.mooreNeighborhoodSFF)
        flattened_probs = probability_matrix.flatten()
        total_prob = flattened_probs.sum()  # normalize to sum to 1
        
        
        if self.verbose >=1:
            print(probability_matrix)
        # if surrounded by walls stay put/stuck
        if total_prob == 0:
            return (0, 0) # No valid move available
        
        flattened_probs /= total_prob

        chosen_index = self.rng.choice(len(flattened_probs), p=flattened_probs)
        move_direction = self.actions.MOORE_ACTIONS[chosen_index]
        
        if self.verbose >=1:
            print(move_direction)
            
        
        
        if move_direction is None:
            return (0, 0)
        return move_direction  # adjust for offset
        
    def update_state(self, new_state:AgentState):
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