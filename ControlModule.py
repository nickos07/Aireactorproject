import numpy as np
from mdptoolbox.mdp import ValueIteration

class ControlModule:
    def __init__(self):
        pass

    @staticmethod
    def generate_P(num_states, probabilities) -> np.ndarray:
        #Generate the transition matrix of probabilities
        p = np.zeros((3, num_states, num_states))
        
        for s in range(num_states):
            #Decrease
            p_d = probabilities[0]
            p[0, s, max(0, s - 2)] += p_d[0]
            p[0, s, max(0, s - 1)] += p_d[1]
            p[0, s, s] += p_d[2]

            #Maintain
            p_m = probabilities[1]
            p[1, s, max(0, s - 1)] += p_m[0]
            p[1, s, s] += p_m[1]
            p[1, s, min(num_states-1, s+1)] += p_m[2]

            #Increase
            p_i = probabilities[2]
            p[2, s, s] += p_i[0]
            p[2, s, min(num_states-1, s+1)] += p_i[1]
            p[2, s, min(num_states-1, s+2)] += p_i[2]
            
        return p

    @staticmethod
    def generate_C(num_states, current_demand):
        #Generate the matrix of costs (3 Actions x 100 States x 100 Next states)
        c = np.zeros((3, num_states, num_states))
        #Array of 100 power levels(from 0.0 to 0.99)
        levels = np.linspace(0, 0.99, num_states)
    
        for a in range(3):  #Loop through the actions
            for s in range(num_states): #Loop through the starting state
                for next_state in range(num_states): #Loop through the state level it can land on
                    #Get power level and calculate demand
                    p_next = levels[next_state]
                    distance = abs(current_demand - p_next)

                    # If demand is below current state, and we increase/maintain high
                    is_moving_away = False

                    if p_next > current_demand and a == 2: #If increase while above demand
                        is_moving_away = True
                    elif p_next < current_demand and a == 0: #If decrease while below demand
                        is_moving_away = True

                    if is_moving_away:
                        c[a, s, next_state] = distance * 2
                    else:
                        c[a, s, next_state] = distance
                
        return c

    @staticmethod
    def solve_control_iteration(current_state, current_demand, p_matrix, discount_factor=0.9):
        #Solve one step in the control loop: deduce best action

        #Get the cost matrix
        c_matrix = ControlModule.generate_C(100, current_demand)
        
        #Use ValueIteration algorithm from pymdptoolbox
        # pass cost matrix as negative because Reward = -Cost
        vi = ValueIteration(p_matrix, -c_matrix, discount_factor)
        
        #Find and return the optimal policy
        vi.run()
        optimal_policy = vi.policy
        return optimal_policy[current_state]

    @staticmethod
    def control_loop(demand, probs, n_states, n_actions, gamma) -> np.ndarray:
        #Run the control loop

        #Generate P
        p_matrix = ControlModule.generate_P(n_states, probs)

        history = np.zeros(len(demand), dtype=np.float64)

        levels = np.linspace(0, 0.99, n_states)
        
        #Start at the initial state in the middle
        current_state = 50
        
        #Iterate through every point in the demand
        for i, current_demand in enumerate(demand):
            #Find optimal action
            optimal_action = ControlModule.solve_control_iteration(
                current_state, current_demand, p_matrix, gamma)
            
            #Get transition probabilities for this action from current state
            transition_probs = p_matrix[optimal_action, current_state, :]
            
            #Determine the next state using numpy.random.choice
            #with the transition probabilities for the chosen action
            next_state = np.random.choice(n_states, p=transition_probs)
            
            #Make sure that state stays within bounds
            next_state = np.clip(next_state, 0, n_states - 1)
            
            #Record the power level of current state in history
            history[i] = levels[current_state]
            
            #Update current state
            current_state = next_state
        
        return history
