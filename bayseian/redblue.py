import numpy as np

# 1. Define the possible states
states = ["Red", "Blue"]
transition_matrix = np.array([
    [0.8, 0.2], 
    [0.3, 0.7]
])

def simulate_markov_process(initial_state, num_steps):
    current_state = initial_state
    red_count = 0
    
    for _ in range(num_steps):
        if current_state == "Red":
            probs = transition_matrix[0]
            red_count += 1
        else:
            probs = transition_matrix[1]
            
        next_state = np.random.choice(states, p=probs)
        current_state = next_state
        
    return red_count

# 3. Run the Simulation
initial_state = "Red"
steps = 15
count=0
iterations = 1000
for _ in range(iterations):
    count+= simulate_markov_process(initial_state, steps)
redprob= count/(steps*iterations)
print(f"Estimated probability of being in Red after {steps} steps: {redprob:.4f}")