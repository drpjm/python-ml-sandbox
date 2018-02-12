# Loads and runs a trained NN policy for controlling the CartPole
import gym
import numpy as np
import tensorflow as tf

env = gym.make("CartPole-v0")
obsv = env.reset()

nn_model_path = "./my_policy_net_pg"

n_max_steps = 1000
with tf.Session() as sess:
    saver = tf.train.Saver()
    # Load up the prior trained NN policy
