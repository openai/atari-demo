import os
import argparse
import numpy as np 
import gym
import tensorflow as tf
from policies import GRUPolicy
from wrappers import wrap_deepmind_npz_sticky, wrap_deepmind_npz_epsilon

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--game', type=str, default='MontezumaRevenge')
parser.add_argument('-d', '--save_dir', type=str, default=None)
parser.add_argument('-n', '--demo_nr',  type=str, default=0)
args = parser.parse_args()

### assumes demo is under `demos/framestack`
if args.save_dir is None:
	save_dir = os.path.join(os.getcwd(), 'demos/npz')
else:
	save_dir = args.save_dir

if not os.path.exists(save_dir):
	os.makedirs(save_dir)

npz_file_name = os.path.join(save_dir, args.game +'_' + str(args.demo_nr) + '.npz')

print(npz_file_name)
assert os.path.exists(npz_file_name)
dat = np.load(npz_file_name)

acs = dat['actions']
obs = dat['observations'][:len(acs)]

env = wrap_deepmind_npz_sticky(gym.make(args.game + 'NoFrameskip-v4'))
#env = wrap_deepmind_npz_epsilon(gym.make(args.game + 'NoFrameskip-v4'))

sess = tf.InteractiveSession()
ac_space = env.action_space
# hack to pass in ob_space
ob_space = obs[0]
nsteps = len(acs)
nbatch = nsteps
num_iters = 100
LR = 0.0003

# create policies
policy_train = GRUPolicy(sess=sess, ob_space=ob_space, 
				ac_space=ac_space, nbatch=nbatch, 
				nsteps=nsteps, memsize=1024, reuse=False)
policy_step = GRUPolicy(sess=sess, ob_space=ob_space,
                        ac_space=ac_space, nbatch=1,
                        nsteps = 1, memsize=1024, reuse=True, deterministic=True)


A = tf.placeholder(tf.int32, [nbatch])
pi = policy_train.pi
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=A, logits=pi)
loss = tf.reduce_mean(loss)

params = tf.trainable_variables()

for p in params:
    loss += 0.00001*tf.reduce_sum(tf.square(p))

grads = tf.gradients(loss, params)
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
        #print('Iteration %d and loss %f' % (i, ls))
    print('Loss %f' % ls)

mask = np.zeros(nbatch)
num_eval = 10000

for rep in range(10000):
    ### optimize with num_iters iterations
    optimize(policy_train, obs, acs, mask, num_iters)

    #### evaluation by taking num_eval steps
    reward_sum = 0
    for testrep in range(10):
        ob = env.reset()
        state = policy_step.initial_state
        for i in range(num_eval):
            if len(ob.shape)<4:
                ob = np.expand_dims(ob, 0)
            a, _, state, _ = policy_step.step(ob, state, [0])
            ob, reward, done, info = env.step(a)
            reward_sum += reward

            if done:
                break

    print('avg return', reward_sum/10)


