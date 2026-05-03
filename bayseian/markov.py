import numpy as np

# 1. Define the possible states
states = ["Red", "Blue"]

# 2. Define the Transition Matrix
# Row 0 (Red): 80% chance to stay Red, 20% chance to move to Blue
# Row 1 (Blue): 30% chance to move to Red, 70% chance to stay Blue
transition_matrix = np.array([
    [0.8, 0.2], 
    [0.3, 0.7]
])

def simulate_markov_process(initial_state, num_steps):
    current_state = initial_state
    state_sequence = [current_state]
    
    for _ in range(num_steps):
        # Determine which row of the matrix to use based on the current state
        if current_state == "Red":
            probs = transition_matrix[0]
        else:
            probs = transition_matrix[1]
            
        # The core Markov decision: pick the next state based on the probabilities
        next_state = np.random.choice(states, p=probs)
        
        # Update for the next iteration
        state_sequence.append(next_state)
        current_state = next_state
        
    return state_sequence

# 3. Run the Simulation
initial_state = "Red"
steps = 15
sequence = simulate_markov_process(initial_state, steps)

# 4. Output the results
print(f"Simulation of {steps} steps starting from {initial_state}:")
print(" -> ".join(sequence))

# Optional: Count occurrences to see the distribution
red_count = sequence.count("Red")
blue_count = sequence.count("Blue")
print(f"\nFinal Tally: Red: {red_count}, Blue: {blue_count}")