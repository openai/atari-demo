import os
import argparse
import numpy as np
import gym
import pickle
import imageio

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--game', type=str, default='MontezumaRevenge')
parser.add_argument('-y', '--screen_height', type=int, default=840)
parser.add_argument('-d', '--save_dir', type=str, default=None)
parser.add_argument('-n', '--demo_nr', type=int, default=0)
parser.add_argument('-f', '--frame_rate', type=int, default=300)
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
demo_actions = dat['actions']

env = gym.make(args.game + 'NoFrameskip-v4')
observations = [env.reset()]

sum_r = 0.
for action in demo_actions:
    obs, reward, done, info = env.step(action)
    observations.append(obs)
    sum_r += reward

print('processed %s demo with %d score' % (args.game, sum_r))

# save video
def process_frame(frame):
    f_out = np.zeros((224, 160, 3), dtype=np.uint8)
    f_out[7:-7, :] = np.cast[np.uint8](frame)
    return f_out
videofile_name = os.path.join(save_dir, '%s_%d.mp4' % (args.game, sum_r))
video_writer = imageio.get_writer(videofile_name, mode='I', fps=60)
for i, obs in enumerate(observations):
    if (i*60) % args.frame_rate == 0:
        video_writer.append_data(process_frame(obs))
video_writer.close()
