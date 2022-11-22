#***Recollecting sample environment***
import numpy as np
#Define states
location_to_state = {
    'L1': 0,
    'L2': 1,
    'L3': 2,
    'L4': 3,
    'L5': 4,
    'L6': 5,
    'L7': 6,
    'L8': 7,
    'L9': 8,
}
#Define actions
actions = [0,1,2,3,4,5,6,7,8]
#Define rewards
rewards = np.array([[0,1,0,0,0,0,0,0,0],
                    [1,0,1,0,1,0,0,0,0],
                    [0,1,0,0,0,1,0,0,0],
                    [0,0,0,0,0,0,1,0,0],
                    [0,1,0,0,0,0,0,1,0],
                    [0,0,1,0,0,0,0,0,0],
                    [0,0,0,1,0,0,0,1,0],
                    [0,0,0,0,1,0,1,0,1],
                    [0,0,0,0,0,0,0,1,0]])

#Mapping indices to location
state_to_location = dict((state,location) for location,state in location_to_state.items())
#Initialize parameters
gamma = 0.75 #Discount factor
alpha = 0.9  #Learning rate

#***Define a function get_optimal_route()***
def get_optimal_route(start_location, end_location):
    # Copy the rewards matrix to new matrix
    rewards_new = np.copy(rewards)

    # Get the ending state corresponding to the ending location as given
    ending_state = location_to_state[end_location]

    # Set the priority of the given ending state to the highest one
    rewards_new[ending_state, ending_state] = 999

    # -----------Q-Learning algorithm-----------

    # Initializing Q-Values
    Q = np.array(np.zeros([9, 9]))

    # Q-Learning process
    for i in range(100000):
        current_state = np.random.randint(0,9)
        # For traversing through the neighbor locations in the maze
        playable_actions = []
        # Iterate through the new rewards matrix and get the actions > 0
        for j in range(9):
            if rewards_new[current_state, j] > 0:
                playable_actions.append(j)
        # Pick an action randomly from the list of playable actions leading to the next state
        next_state = np.random.choice(playable_actions)
        # Compute the temporal difference
        TD = rewards_new[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[
            current_state, next_state]
        # Update the Q-Value using the Bellman equation
        Q[current_state, next_state] += alpha * TD
    # Initialize the optimal route with the starting location
    route = [start_location]
    # Initialize with the value of starting location
    next_location = start_location
    while (next_location != end_location):
        # Fetch the starting state
        starting_state = location_to_state[start_location]
        # Fetch the highest Q-value pertaining to starting state
        next_state = np.argmax(Q[starting_state,])
        next_location = state_to_location[next_state]
        route.append(next_location)
        # Update the starting location for the next iteration
        start_location = next_location
    return route
print(get_optimal_route('L4','L6'))
print(get_optimal_route('L1','L6'))