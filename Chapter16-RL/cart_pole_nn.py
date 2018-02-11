# Demo of a naive cart-pole control algorithm
import gym
import numpy as np
import tensorflow as tf

# This example code is based on the REINFORCE policy gradient algorithm.

################################################
# Construct the NN structure with tensorflow
################################################
n_inputs = 4
n_hidden = 4 # Four hidden units
n_outputs = 1 # Only output is a probability: 1 -> action = 0; 0 -> action = 1

initializer = tf.contrib.layers.variance_scaling_initializer()

learning_rate = 0.01

# Neural net structure: 4 inputs fully, 1 fully connected layer, to one output
X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden_layer = tf.layers.dense(X, n_hidden, activation=tf.nn.elu,
                                kernel_initializer=initializer)
logits = tf.layers.dense(hidden_layer, n_outputs, kernel_initializer=initializer)
outputs = tf.nn.sigmoid(logits)

# Use outputs as probability to choose left (0) or right (1) acceleration
prob_left_right = tf.concat(axis=1, values=[outputs, 1-outputs])
action = tf.multinomial(tf.log(prob_left_right), num_samples=1)

y = 1. - tf.to_float(action)
# Compute entropy between the logit output and the chosen action
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)

optimizer = tf.train.AdamOptimizer(learning_rate)
# Just compute - do not apply gradients yet!
grads_and_vars = optimizer.compute_gradients(cross_entropy)
# List comprehension to load the gradients from the optimizer's computation
gradients = [grad for grad, variable in grads_and_vars]
gradient_placeholders = []
grads_and_vars_feed = []

for grad, variable in grads_and_vars:
    gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))

training_operation = optimizer.apply_gradients(grads_and_vars_feed)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

#################################################
# Create execution framework and helper functions
#################################################

def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.empty(len(rewards))
    cumulative_rewards = 0
    for step in reversed( range(len(rewards)) ):
        cumulative_rewards = rewards[step] + (cumulative_rewards*discount_rate)
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discout):
	all_discounted_rewards = [discount_rewards(rewards, discount_rate)
							for rewards in all_rewards]
	flat_rewards = np.concatenate(all_discounted_rewards)
	reward_mean = flat_rewards.mean()
	reward_std = flat_rewards.std()
	return [(discounted_rewards - reward_mean) / reward_std
			for discounted_rewards in all_discounted_rewards]
