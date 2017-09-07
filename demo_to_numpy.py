import os
import argparse
import numpy as np
import gym
import pickle
from wrappers import wrap_deepmind_npz

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--game', type=str, default='MontezumaRevenge')
parser.add_argument('-d', '--save_dir', type=str, default=None)
parser.add_argument('-n', '--demo_nr', type=int, default=0)
parser.add_argument('-fs', '--frame_skip', type=int, default=4)
args = parser.parse_args()

if args.save_dir is None:
    save_dir = os.path.join(os.getcwd(), 'demos')
else:
    save_dir = args.save_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
demo_file_name = os.path.join(save_dir, args.game + str(args.demo_nr) + '.demo')
assert os.path.exists(demo_file_name)
with open(demo_file_name, "rb") as f:
    dat = pickle.load(f)
demo_actions = dat['actions'][::args.frame_skip]

env = wrap_deepmind_npz(gym.make(args.game + 'NoFrameskip-v4'))
observations = [env.reset()]

sum_r = 0.
for action in demo_actions:
    obs, reward, done, info = env.step(action)
    observations.append(obs)
    sum_r += reward

print('processed %s demo with %d score' % (args.game, sum_r))

# save to numpy
obs = np.stack(observations, axis=0)
act = np.asarray(demo_actions)
file_name = os.path.join(save_dir, '%s_%d.npz' % (args.game, sum_r))
np.savez(file_name, observations=obs, actions=act)
