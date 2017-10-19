import argparse
import numpy as np 
import gym
import tensorflow as tf
from atari_demo.policies import GRUPolicy
from atari_demo.wrappers import wrap_deepmind
import imageio
import os.path as osp
from atari_demo.utils import load_as_pickled_object
import os

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--game', type=str, default='MontezumaRevenge')
parser.add_argument('-d', '--demo_dir', type=str, default=osp.join(osp.expanduser('~'),'data/rcall/gce/atari-demo/'))
parser.add_argument('-s', '--save_dir', type=str, default=os.path.join(os.getcwd(), 'demos'))
parser.add_argument('-n', '--demo_nr',  type=str, default=0)
parser.add_argument('--nenv', type=int, default=64)
parser.add_argument('--memsize', type=int, default=3200)
parser.add_argument('-f', '--frame_rate', type=int, default=300)
parser.add_argument('--ngpu', type=int, default=8)
parser.add_argument('--time_steps', type=int, default=128)
parser.add_argument('--max_grad_norm', type=float, default=1.)
parser.add_argument('--noop_reset', dest='noop_reset', action='store_true')
parser.add_argument('--sticky_actions', dest='sticky_actions', action='store_true')
parser.add_argument('--epsilon_greedy', dest='epsilon_greedy', action='store_true')
parser.add_argument('--no_stuck', dest='no_stuck', action='store_true')
args = parser.parse_args()

# eval env
env_id = args.game+'NoFrameskip-v4'
nenv_test = 1
env = gym.make(env_id)
env = wrap_deepmind(env, noop_reset=args.noop_reset, sticky_actions=args.sticky_actions, epsilon_greedy=args.epsilon_greedy)
obs = env.reset()
done = False

sess = tf.InteractiveSession()
ac_space = env.action_space
ob_space = env.observation_space
policy_step = GRUPolicy(sess=sess, ob_space=ob_space,
                        ac_space=ac_space, nbatch=1,
                        nsteps=1, memsize=args.memsize, reuse=False, deterministic=True)
test_state = policy_step.initial_state

params = tf.trainable_variables()
saver = tf.train.Saver(params)
saver.restore(sess, osp.join(os.getcwd(), args.game + '_new_small_params-999'))

def get_action(obs, done, state):
    feed_dict = {policy_step.X: obs.reshape(policy_step.X.shape)}
    feed_dict.update({policy_step.M: done*np.ones((1,))})
    feed_dict.update({policy_step.S: state.reshape(policy_step.S.shape)})
    a, snew = sess.run([policy_step.a, policy_step.snew], feed_dict=feed_dict)
    return a, snew

class ActionState(object):
    def __init__(self, rambytes, action):
        self.rambytes = rambytes
        self.action = action
    def __hash__(self):
        return hash(self.rambytes)
    def __eq__(self, other):
        return (self.rambytes==other.rambytes and self.action==other.action)

prev_actions = []
actions = []
rewards = []
a, test_states = get_action(obs, 1., test_state)
while not done:
    if args.no_stuck:
        rambytes = env.unwrapped._get_ram().tostring()
        state_act = ActionState(rambytes, a)
        counter = 0
        while state_act in prev_actions and counter<18:
            a = (a+1) % 18
            state_act = ActionState(rambytes, a)
            counter += 1
        prev_actions.append(state_act)
    obs, rew, done, info = env.step(a)
    actions.append(a)
    rewards.append(rew)
    a, test_state = get_action(obs, 0., test_state)

print('achieved %d score' % sum(rewards))

env = gym.make(args.game + 'NoFrameskip-v4')
observations = [env.reset()]

sum_r = 0.
for action in actions:
    for skip in range(4):
        obs, reward, done, info = env.step(action)
        observations.append(obs)
        sum_r += reward

print('processed %s demo with %d score' % (args.game, sum_r))


if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# save video
def process_frame(frame):
    f_out = np.zeros((224, 160, 3), dtype=np.uint8)
    f_out[7:-7, :] = np.cast[np.uint8](frame)
    return f_out
videofile_name = os.path.join(args.save_dir, '%s_%d.mp4' % (args.game, sum_r))
video_writer = imageio.get_writer(videofile_name, mode='I', fps=60)
for i, obs in enumerate(observations):
    if (i*60) % args.frame_rate == 0:
        video_writer.append_data(process_frame(obs))
video_writer.close()

