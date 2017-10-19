import argparse
import numpy as np 
import gym
import tensorflow as tf
from atari_demo.policies import GRUPolicy
from atari_demo.wrappers import wrap_deepmind_no_frameskip
from atari_demo.cloned_vec_env import make_cloned_vec_env
from rl_algs.common.vec_env.subproc_vec_env import SubprocVecEnv
from rl_algs import bench, logger
import os.path as osp
from atari_demo.utils import load_as_pickled_object
import os

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--game', type=str, default='MontezumaRevenge')
parser.add_argument('-d', '--demo_dir', type=str, default=osp.join(osp.expanduser('~'),'data/rcall/gce/atari-demo/'))
parser.add_argument('-n', '--demo_nr',  type=str, default=0)
parser.add_argument('--nenv', type=int, default=64)
parser.add_argument('--memsize', type=int, default=3200)
parser.add_argument('--ngpu', type=int, default=8)
parser.add_argument('--time_steps', type=int, default=128)
parser.add_argument('--max_grad_norm', type=float, default=1.)
parser.add_argument('--noop_reset', dest='noop_reset', action='store_true')
parser.add_argument('--sticky_actions', dest='sticky_actions', action='store_true')
parser.add_argument('--epsilon_greedy', dest='epsilon_greedy', action='store_true')
parser.add_argument('--no_stuck', dest='no_standstill', action='store_true')
args = parser.parse_args()

# load demo
fname = osp.join(args.demo_dir, args.game + str(args.demo_nr) + '_extended.pkl')
dat = load_as_pickled_object(fname)
best_action_dict = dat['best_action_dict']
possible_actions_dict = dat['possible_actions_dict']
print('Loaded %d states and %d actions' % (len(possible_actions_dict),len(best_action_dict)))
n_actions = max([max(s) for s in possible_actions_dict.values()])+1

# train envs
env_id = args.game+'NoFrameskip-v4'
train_env = make_cloned_vec_env(args.nenv, env_id, possible_actions_dict, best_action_dict, wrap_deepmind_no_frameskip, mode='available_actions')
train_env.reset()

# eval envs
nenv_test = args.nenv // 2
def make_env(rank):
    def env_fn():
        env = gym.make(env_id)
        env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), str(rank)))
        return wrap_deepmind_no_frameskip(env, noop_reset=args.noop_reset, sticky_actions=args.sticky_actions, epsilon_greedy=args.epsilon_greedy)
    return env_fn
test_env = SubprocVecEnv([make_env(i) for i in range(nenv_test)])
test_obs = test_env.reset()
test_dones = np.zeros(nenv_test)

sess = tf.InteractiveSession()
ac_space = train_env.action_space
ob_space = train_env.observation_space
nbatch_train = args.time_steps*args.nenv // args.ngpu

# create policies
with tf.device('/gpu:0'):
    policy_train = [GRUPolicy(sess=sess, ob_space=ob_space,
                    ac_space=ac_space, nbatch=nbatch_train,
                    nsteps=args.time_steps, memsize=args.memsize, reuse=False, drop_prob=0.2)]
for i in range(1,args.ngpu):
    with tf.device('/gpu:%d' % i):
        policy_train.append(GRUPolicy(sess=sess, ob_space=ob_space,
                                  ac_space=ac_space, nbatch=nbatch_train,
                                  nsteps=args.time_steps, memsize=args.memsize, reuse=True, drop_prob=0.2))
train_states = [p.initial_state for p in policy_train]

policy_step = []
for i in range(args.ngpu):
    with tf.device('/gpu:%d' % i):
        policy_step.append(GRUPolicy(sess=sess, ob_space=ob_space,
                        ac_space=ac_space, nbatch=nenv_test//args.ngpu,
                        nsteps=1, memsize=args.memsize, reuse=True, deterministic=True))
test_states = [p.initial_state for p in policy_step]

def possible_action_cross_ent(logits, labels):
    return tf.reduce_logsumexp(logits, axis=1) - tf.reduce_logsumexp(tf.where(labels>0, logits, -99999999.*tf.ones_like(logits)), axis=1)

# get training loss function
A = [tf.placeholder(tf.int32, [nbatch_train,n_actions]) for i in range(args.ngpu)]
M = [tf.placeholder(tf.float32, [nbatch_train]) for i in range(args.ngpu)]
loss = sum([tf.reduce_mean(possible_action_cross_ent(logits=poli.pi, labels=Ai) * (1. - Mi))
                       for Ai,Mi,poli in zip(A,M,policy_train)]) / args.ngpu

params = tf.trainable_variables()
saver = tf.train.Saver(params)
saver.restore(sess, osp.join(os.getcwd(), args.game + '_new_small_params-999'))

for p in params:
    loss += 3e-6 * tf.reduce_sum(tf.square(p))

LR = tf.placeholder(tf.float32, shape=())
grads = tf.gradients(loss, params)
if args.max_grad_norm is not None:
    grads, _grad_norm = tf.clip_by_global_norm(grads, args.max_grad_norm)
grads = list(zip(grads, params))
trainer = tf.train.AdamOptimizer(learning_rate=LR)
_train = trainer.apply_gradients(grads)

sess.run(tf.global_variables_initializer())

def optimize(obs, acs, acs_mask, dones, states, lr):
    obs, acs, acs_mask, dones = [np.split(x,args.ngpu) for x in (obs, acs, acs_mask, dones)]
    feed_dict = {p.X:o for p,o in zip(policy_train,obs)}
    feed_dict.update({Ai: a.reshape((-1, n_actions)) for Ai, a in zip(A, acs)})
    feed_dict.update({Mi: m.flatten() for Mi, m in zip(M, acs_mask)})
    feed_dict.update({p.M:d for p,d in zip(policy_train,dones)})
    feed_dict.update({p.S:s for p,s in zip(policy_train,states)})
    feed_dict.update({LR: lr})
    res = sess.run([_train, loss] + [p.snew for p in policy_train], feed_dict=feed_dict)
    ls = res[1]
    snew = res[2:]
    return ls, snew

def get_action(obs, dones, states):
    obs, dones = [np.split(x, args.ngpu) for x in (obs, dones)]
    feed_dict = {p.X: o for p, o in zip(policy_step, obs)}
    feed_dict.update({p.M: d for p, d in zip(policy_step, dones)})
    feed_dict.update({p.S: s for p, s in zip(policy_step, states)})
    res = sess.run([p.a for p in policy_step] + [p.snew for p in policy_step], feed_dict=feed_dict)
    a = res[:args.ngpu]
    snew = res[args.ngpu:]
    return a, snew

test_rets = [0.]
start_lr = 3e-4
for epoch in range(99999):
    lr = start_lr * (max([1000 - epoch,1])/1000.)

    for i in range(10):
        # get data from train_env
        obs, rews, dones, best_actions, action_masks = train_env.step(args.time_steps)

        # optimize for one step
        train_loss, train_states = optimize(obs, best_actions, action_masks, dones, train_states, lr)

    # evaluate
    for i in range(args.time_steps):
        a, test_states = get_action(test_obs, test_dones, test_states)
        test_obs, _, test_dones, infos = test_env.step(np.concatenate(a))
        for info in infos:
            if 'episode' in info:
                test_rets.append(info['episode']['r'])
                if len(test_rets) > nenv_test:
                    test_rets.pop(0)

    logger.log('completed epoch %d with loss %.4f and running mean test return %d' % (epoch, train_loss, sum(test_rets)/len(test_rets)))

    if (epoch+1) % 100 == 0:
        saver.save(sess, osp.join(os.getcwd(), args.game + '_new_small_params'), global_step=epoch)

