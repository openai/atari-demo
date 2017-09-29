import os
import argparse
import numpy as np
import gym
import pickle
from wrappers import ResetWrapper, FastSkipEnv

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--game', type=str, default='MontezumaRevenge')
parser.add_argument('-d', '--save_dir', type=str, default=None)
parser.add_argument('-n', '--demo_nr', type=int, default=0)
parser.add_argument('-s', '--frame_skip', type=int, default=4)
parser.add_argument('-q', '--search_depth', type=int, default=8)
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
n = len(demo_actions)

env = ResetWrapper(FastSkipEnv(gym.make(args.game + 'NoFrameskip-v4'), args.frame_skip))
env.reset()
nr_actions = env.action_space.n
print('game has %d actions' % nr_actions)

states = [env.clone_state()]
sum_reward = 0
for i,action in enumerate(demo_actions):
    ob, reward, done, info = env.step(action)
    sum_reward += reward
    states.append(env.clone_state())
print('return achieved in parsed demo was %d' % sum_reward) # check to see nothing is going wrong

state_value_depth_dict = {states[-1]: (1,9999999)} # final state has value of 1 and has been 'searched' to infinite depth
state_best_action_dict = {} # to hold targets for behavior cloning
state_possible_actions_dict = {} # to hold all actions for reconstructing all possible sequences that end in the same final state

def search_states(state, steps_remaining):
    # traverse the tree starting from this state
    # output value of state = discounted probability of ending up in the final state of the demo, under epsilon greedy policy
    # save best found action for behavior cloning

    # have we already searched this state?
    if state in state_value_depth_dict:
        val, depth = state_value_depth_dict[state]
        if depth >= steps_remaining: # keep going if we already visited this state but did not yet search deeply
            return val, depth
    elif steps_remaining==0:
        return 0,0 # if we don't see a path between this state and the final state we'll pessimistically evaluate the value at zero

    # traverse children of this state
    results = []
    for a in range(nr_actions):
        env.restore_state(state)
        env.step(a)

        if env.game_over: # if game-over we know for sure this state does not connect to the final state of the demo
            val = 0
            dep = 99999999
        else: # recursively search children of this child state
            newstate = env.clone_state()
            val, dep = search_states(newstate, steps_remaining-1)

        results.append((val, dep, a))

    # calculate value of the current state and save best action for behavior cloning
    sorted_results = sorted(results)
    maxval, _, best_action = sorted_results[-1]
    value_this_state = 0.99*maxval + 0.01*sum([s[0] for s in results])/nr_actions # epsilon greedy policy with eps=0.01
    value_this_state *= 0.999 # discounting
    depth_searched = min([s[1] for s in results]) + 1
    state_value_depth_dict[state] = (value_this_state, depth_searched)

    minval = sorted_results[0][0]
    if maxval>minval: # only save best action for cloning if it is actually better than any other action
        state_best_action_dict[state] = best_action

    # save all actions that we know keep us on the right trajectory, for generating random sequences te learn on
    possible_actions = [s[2] for s in results if s[0]>0]
    if len(possible_actions)>0:
        state_possible_actions_dict[state] = possible_actions

    return value_this_state, depth_searched


for stepnr in reversed(range(len(states))):
    val, dep = search_states(states[stepnr], args.search_depth)
    print('working back %d steps, now at step %d. demo state has value %.4e, searched to depth %d' % (len(states), stepnr, val, dep))

print('processed demo of %d timesteps and extracted %d obs,action pairs' % (len(demo_actions), len(state_best_action_dict)))

# save extracted obs,action pairs for cloning
states, best_actions = zip(*state_best_action_dict.items())
observations = np.array([s.ram for s in states])
best_actions = np.array(best_actions)
extended_demo_file_name = os.path.join(save_dir, args.game + str(args.demo_nr) + '_extended.npz')
np.savez(extended_demo_file_name, observations=observations, best_actions=best_actions)

# pickle both action dicts
extended_demo_file_name = os.path.join(save_dir, args.game + str(args.demo_nr) + '_extended.pkl')
with open(extended_demo_file_name, "wb") as f:
    pickle.dump({'state_best_action_dict': state_best_action_dict, 'state_possible_actions_dict': state_possible_actions_dict}, f)

