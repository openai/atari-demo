import os
import argparse
import numpy as np 
import gym
from collections import deque
import pickle
import imageio
import tensorflow as tf
from policies import CnnPolicy, GRUPolicy, LstmPolicy, MlpPolicy
from rl_algs.common.atari_wrappers import wrap_deepmind

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--game', type=str, default='MontezumaRevenge')
parser.add_argument('-d', '--save_dir', type=str, default=None)
parser.add_argument('-n', '--demo_nr',  type=str, default=0)
args = parser.parse_args()

if args.save_dir is None:
	save_dir = os.path.join(os.getcwd(), 'demos/framestack')
else:
	save_dir = args.save_dir

if not os.path.exists(save_dir):
	os.makedirs(save_dir)

npz_file_name = os.path.join(save_dir, args.game +'_' + str(args.demo_nr) + '.npz')

assert os.path.exists(npz_file_name)
dat = np.load(npz_file_name)

obs = dat['observations']
acs = dat['actions']

env = wrap_deepmind(gym.make(args.game + 'NoFrameskip-v4'))

# one_hot encode the actions
acs_size = env.action_space.n
#acs = np.eye(env.action_space.n)[acs]

sess = tf.InteractiveSession()
ac_space = env.action_space
# hack to pass in ob_space
ob_space = obs[0]
nsteps = len(acs)
nbatch = nsteps
num_iters = 10
max_grad_norm = 0.5
LR = 0.001

# create policies
policy_train = GRUPolicy(sess=sess, ob_space=ob_space, 
				ac_space=ac_space, nbatch=nbatch, 
				nsteps=nsteps, reuse=False)
policy_step = GRUPolicy(sess=sess, ob_space=ob_space,
                        ac_space=ac_space, nbatch=1,
                        nsteps = 1, reuse=True)


A = tf.placeholder(tf.int32, [nbatch])
pi = policy_train.pi
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=A, logits=pi)
loss = tf.reduce_mean(loss)

params = tf.trainable_variables()
grads  = tf.gradients(loss, params)

if max_grad_norm is not None:
    grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
    
grads = list(zip(grads, params))
trainer = tf.train.AdamOptimizer(learning_rate=LR)
_train = trainer.apply_gradients(grads)

sess.run(tf.global_variables_initializer())

def optimize(policy, obs, acs, mask, num_iters):
    feed_dict = {policy.X: obs,
                A: acs,
                policy.M: mask,
                policy.S: policy.initial_state}
    
    for i in range(num_iters):
        _, ls = sess.run([_train, loss], feed_dict=feed_dict)
        print('Iteration %d and loss %f' % (i, ls))

    
mask = np.zeros(nbatch)

optimize(policy_train, obs, acs, mask, num_iters)

num_eval = 1000
ob = env.reset()
state = policy_step.initial_state
mask = [0]
reward_sum = 0

obs = deque(maxlen=4)
for i in range(4):
    obs.append(ob)
    
    
def stackframe(obs):
    return np.expand_dims(np.squeeze(np.array(obs)).swapaxes(0, 1).swapaxes(1, 2), 0)


for i in range(num_eval):
    obs_input = stackframe(obs)
    
    for _ in range(4):
        a, _, state, _ = policy_step.step(obs_input, state, mask)
        ob, reward, done, info = env.step(a)
        reward_sum += reward
        
        if done:
            mask = [1]
            state = policy_step.initial_state
            ob = env.reset()

        obs.append(ob)

print('summed reward', reward_sum)
    
    
