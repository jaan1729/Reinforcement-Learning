import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use("ggplot")

SIZE = 10
HM_EPISODES = 25000
MOVE_PENALTY = 1
ENMY_PENALTY = 300
FOOD_REWARD = 25
epsilon = 0.9
EPS_DECAY = 0.9998
SHOW_EVERY = 3000

start_q_table = None

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1
FOOD_N = 2
ENMY_N = 3

d = {1:(255, 175, 0),
     2:(0, 255, 0),
     3:(0, 0, 255)}

class Blob:
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)
    def __str__(self):
        return f"{self.x}, {self.y}"
    
    def __sub__(self, other):
        return (self.x - other.x, self.y- other.y)

    def action(self, choice):
        if choice == 0:
            self.move(x=1, y=1)
        if choice == 0:
            self.move(x=1, y=-1)
        if choice == 0:
            self.move(x=-1, y=-1)
        if choice == 0:
            self.move(x=-1, y=1)
    
    def move(self, x = False, y = False):
        if not x:
            self.x += np.random.randint(-1,2)
        else:
            self.x += x
        if not x:
            self.y += np.random.randint(-1,2)
        else:
            self.y += y
        
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE-1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1

if start_q_table == None:
    q_table = {}
    for x1 in range(-SIZE+1, SIZE):
        for y1 in range(-SIZE+1, SIZE):
            for x2 in range(-SIZE+1, SIZE):
                for y2 in range(-SIZE+1, SIZE):
                    q_table[((x1, y1), (x2, y2))] = np.random.randint(-5,0, size = 4)
else:
    with open(q_table, "rb") as f:
        q_table = pickle.load(f)
episode_rewards = []
for episode in range(HM_EPISODES):
    print("episode no: ", episode)
    player = Blob()
    food = Blob()
    enmy = Blob()

    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon: {epsilon}")
        print(f"{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False
    
    episode_reward = 0
    for i in range(200):
        obs = (player - food), (player - enmy)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0,4)

        player.action(action)

        if player.x == enmy.x and player.y == enmy.y:
            reward = -ENMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY
        
        new_obs = (player - food, player - enmy)
        future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        elif reward == -ENMY_PENALTY:
            new_q = -ENMY_PENALTY
        else:
            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * future_q)
        
        q_table[obs][action] = new_q

        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype = np.int)
            env[food.x][food.y] = d[FOOD_N]
            env[player.x][player.y] = d[PLAYER_N]
            env[enmy.x][enmy.y] = d[ENMY_N]

            img = Image.fromarray(env, "RGB")
            img = img.resize((600,600))
            cv2.imshow("", np.array(img))
            
            if reward == FOOD_REWARD or reward == -ENMY_PENALTY:
                if cv2.waitKey(500) & 0xFF == ord("q"):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            episode_reward += reward
            if reward == FOOD_REWARD or reward == -ENMY_PENALTY:
                break

        episode_rewards.append(episode_reward)
        epsilon *= EPS_DECAY

    moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode = "valid")

    plt.plot([i for i in range(len(moving_avg))], moving_avg)
    plt.ylabel(f"reward {SHOW_EVERY}ma")
    plt.xlabel("episode #")
    plt.show() 

    with open(f"qtable-{int(time.time())}.pikle", "wb") as f:
        pickle.dump(q_table, f)



