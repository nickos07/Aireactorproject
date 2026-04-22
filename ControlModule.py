# Import required dependencies
import numpy as np
import mdptoolbox as mdp

#gilipollas

class ControlModule:
    def __init__(self):
        """ Dummy constructor to use the Python Class as a namespace """
        pass

    @staticmethod
    def generate_P(num_states, probabilities) -> np.ndarray:
        """ Function that generates the probabilities (transition) matrix """
        P = np.zeros((3, num_states, num_states))
        
        for s in range(num_states):
            # Action: Decrease (d) -> Outcomes: -2, -1, 0
            p_d = probabilities['decrease'] # [0.55, 0.20, 0.25] -> -2, -1, 0 [cite: 250]
            P[0, s, max(0, s - 2)] += p_d[0]
            P[0, s, max(0, s - 1)] += p_d[1]
            P[0, s, s]             += p_d[2]

            # Action: Maintain (m) -> Outcomes: -1, 0, +1
            p_m = probabilities['maintain'] # [0.95, 0.025, 0.025] -> -1, 0, +1 [cite: 256]
            P[1, s, max(0, s - 1)]          += p_m[0]
            P[1, s, s]                      += p_m[1]
            P[1, s, min(num_states-1, s+1)] += p_m[2]

            # Action: Increase (i) -> Outcomes: 0, +1, +2
            p_i = probabilities['increase'] # [0.65, 0.25, 0.1] -> 0, +1, +2 [cite: 257]
            P[2, s, s]                      += p_i[0]
            P[2, s, min(num_states-1, s+1)] += p_i[1]
            P[2, s, min(num_states-1, s+2)] += p_i[2]
            
            return P

    @staticmethod
    def generate_C(num_states, current_demand):
    """
    Generates matrix C (Actions x States x States)
    current_demand: float between 0 and 1
    """
        C = np.zeros((3, num_states, num_states))

    # Power levels (lower bounds of intervals)
        levels = np.linspace(0, 0.99, num_states) # 0.0, 0.01, ..., 0.99 [cite: 73]
    
        for a in range(3):
            for s in range(num_states):
                for s_next in range(num_states):
                    # Target power at destination
                    p_next = levels[s_next]
                    distance = abs(current_demand - p_next) [cite: 118]
                
                    # Penalization logic [cite: 119, 122]
                    # If demand is below current state and we increase/maintain high
                    is_moving_away = False
                    if p_next > current_demand and a == 2: # Increasing while above demand
                        is_moving_away = True
                    elif p_next < current_demand and a == 0: # Decreasing while below demand
                        is_moving_away = True
                    
                    C[a, s, s_next] = distance * 2 if is_moving_away else distance
                
        return C

    @staticmethod
    def solve_control_iteration(current_state, current_demand, P_matrix, discount_factor=0.9):
        """
        Solves one step of the control loop.
        Returns the optimal action index (0, 1, or 2).
        """
        # 1. Generate the cost matrix for this specific demand point [cite: 170]
        C_matrix = generate_cost_matrix(100, current_demand)
        
        # 2. Initialize Value Iteration [cite: 174]
        # Note: pymdptoolbox uses Reward (R). Reward = -Cost.
        vi = mdp.ValueIteration(P_matrix, -C_matrix, discount_factor)
        
        # 3. Run the algorithm to find the optimal policy
        vi.run()
        
        # 4. Extract the best action for our current state [cite: 173]
        optimal_policy = vi.policy # Array of 100 optimal actions
        return optimal_policy[current_state]

    @staticmethod
    def control_loop(demand: np.ndarray, 
                     probs: np.ndarray,
                     n_states: np.int32, 
                     n_actions: np.int32,
                     gamma: np.float64) -> np.ndarray:
        """ Function that computes all the required iterations (control-loop) to satisfy the power demand """
        ### TO BE COMPLETED BY THE STUDENTS ###

        ### DUMMY BEHAVIOUR TO PREVENT CRASHING (MUST BE DELETED AFTER THE FULL IMPLEMENTATION) ###
        return np.zeros_like(a=demand, dtype=np.float64)
        ### ###
