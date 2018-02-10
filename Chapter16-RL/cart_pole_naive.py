# Demo of a naive cart-pole control algorithm
import gym
import numpy as np

def naive_policy(obsv):
    # The cart-pole simulator has four states: x, x_dot, theta, theta_dot
    # In the simulator, action = 1 accelerates right, action = 0 accelerates left
    theta = obsv[2]
    # If the angle is positive (clockwise deflection), accelerate right
    # to "get under" the pole
    if theta > 0:
        action = 1
    else:
        action = 0
    return action

totals = []

env = gym.make("CartPole-v0")
# env.render()

# Run the naive policy and compute its average rewards
for episode in range(1000):
    episode_rewards = 0
    obsv = env.reset()
    if episode % 20 == 0:
        print( "Running episode " + str(episode) )
    # 1000 actions per episode
    for i in range(1000):
        action = naive_policy(obsv)
        obsv, reward, is_done, info = env.step(action)
        # env.render()
        episode_rewards += reward
        if is_done:
            break
    totals.append(episode_rewards)

# env.close()

mean_reward = np.mean(totals)
min_reward = np.min(totals)
max_reward = np.max(totals)

print("Mean Reward = " + str(mean_reward))
print("Maximum iterations (max reward) = " + str(max_reward))
