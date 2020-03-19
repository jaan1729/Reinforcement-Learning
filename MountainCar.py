import gym
import matplotlib
import numpy as np

env = gym.make("MountainCar-v0")

DESCRETE_OS_SIZE = [20] * len(env.observation_space.high)
descrete_state_window_size = (env.observation_space.high - env.observation_space.low) / DESCRETE_OS_SIZE
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 5000
epsilon = 0.5
START_EPSILON_VALUE = 1
END_EPSILON_VALUE = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_VALUE - START_EPSILON_VALUE)

def get_descrete_state(state):
    descrete_state = (state - env.observation_space.low) / descrete_state_window_size
    return tuple(descrete_state.astype(int)) 

q_table = np.random.uniform(low = -2, high = 0, size = (DESCRETE_OS_SIZE + [env.action_space.n]))


for episode in range(EPISODES):
    if True:
        print(episode)
        render = True
    else:
        render = False
    
    descrete_state = get_descrete_state(env.reset())
    done = False
    while not done:
        action = np.argmax(q_table[descrete_state])
        new_state, reward, done, _ = env.step(action)
        if render:
            env.render()
        new_descrete_state = get_descrete_state(new_state)
        if not done:
            current_q = q_table[descrete_state + (action,)]
            future_q = np.max(q_table[new_descrete_state])
            updated_q = (1- LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * future_q)
            q_table[descrete_state + (action,)] = updated_q
        elif new_state[0] > env.goal_position:
            print("we are done at", episode)
            q_table[descrete_state + (action,)] = 0
        descrete_state = new_descrete_state

        if END_EPSILON_VALUE >= episode >= START_EPSILON_VALUE:
            epsilon-=epsilon_decay_value
env.close()