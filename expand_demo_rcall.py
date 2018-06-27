import os
import argparse
import gym
import pickle
import numpy as np
from rl_algs import logger
from rcall.gce import call

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--game', type=str, default='MontezumaRevenge')
parser.add_argument('-d', '--save_dir', type=str, default='/root/code/atari-demo/demos')
parser.add_argument('-n', '--demo_nr', type=int, default=0)
parser.add_argument('-s', '--frame_skip', type=int, default=4)
parser.add_argument('-q', '--search_depth', type=int, default=8)
args = parser.parse_args()


def expand_demo(my_nr, nr_workers):
    from atari_demo.wrappers import ResetWrapper, FastSkipEnv

    demo_file_name = os.path.join(args.save_dir, args.game + str(args.demo_nr) + '.demo')
    assert os.path.exists(demo_file_name)

    with open(demo_file_name, "rb") as f:
        dat = pickle.load(f)
    demo_actions = dat['actions']

    # chop up the demo in nr_workers pieces
    n = len(demo_actions)
    steps_per_worker = n / nr_workers
    first_nr = int(np.ceil(my_nr * steps_per_worker))
    last_nr = int(np.ceil((my_nr+1) * steps_per_worker))

    env = ResetWrapper(FastSkipEnv(gym.make(args.game + 'NoFrameskip-v4'),1))
    env.reset()
    nr_actions = env.action_space.n
    print('game has %d actions' % nr_actions)
    demo_states = [env.clone_state()]
    sum_reward = 0
    for i,action in enumerate(demo_actions):
        ob, reward, done, info = env.step(action)
        sum_reward += reward
        demo_states.append(env.clone_state())
    print('return achieved in parsed demo was %d' % sum_reward) # check to see nothing is going wrong

    last_ind = min([len(demo_states), last_nr+int(steps_per_worker)])
    demo_states = demo_states[:last_ind]

    state_value_depth_dict = {s.rambytes: (1e300,99999999) for s in demo_states[-args.frame_skip:]} # initial state values
    demo_values_thus_far = [1e300]
    state_best_action_dict = {} # to hold targets for behavior cloning
    m = min([len(demo_actions),len(demo_states)])
    state_possible_actions_dict = {s.rambytes:{a} for s,a in zip(demo_states[:m],demo_actions[:m])} # to hold all actions for reconstructing all possible sequences that end in the same final state

    def search_states(state, steps_remaining):
        # traverse the tree starting from this state
        # output value of state = discounted probability of ending up in the final state of the demo, under epsilon greedy policy
        # save best found action for behavior cloning

        # have we already searched this state?
        if state.rambytes in state_value_depth_dict:
            val, depth = state_value_depth_dict[state.rambytes]
            if depth >= steps_remaining: # keep going if we already visited this state but did not yet search deeply
                return val, depth
        elif steps_remaining==0:
            return 0,0 # if we don't see a path between this state and the final state we'll pessimistically evaluate the value at zero

        # traverse children of this state
        results = []
        state_action_pairs_list = []
        final_ram_states = []
        for a in range(nr_actions):
            env.restore_state(state)
            state_action_pairs_list.append([])

            newstate = state
            for step in range(args.frame_skip):
                state_action_pairs_list[a].append((newstate.rambytes, a))
                env.step(a)
                newstate = env.clone_state()
                if newstate.game_over:
                    break

            if newstate.game_over:  # if game-over we know for sure this state does not connect to the final state of the demo
                val = 0
                dep = 99999999
            else:  # recursively search children of this child state
                val, dep = search_states(newstate, steps_remaining - 1)

            results.append((val, dep, a))
            final_ram_states.append(newstate.rambytes)

        # connect up the states an keep track of actions that keep us on course
        for step in reversed(range(args.frame_skip)):
            for a in range(nr_actions):
                state_action_pairs = state_action_pairs_list[a]

                is_connected = False
                if step == args.frame_skip-1:
                    if final_ram_states[a] in state_possible_actions_dict:
                        is_connected = True
                elif state_action_pairs[step+1][0] in state_possible_actions_dict:
                    is_connected = True

                if is_connected:
                    si,ai = state_action_pairs[step]
                    if si not in state_possible_actions_dict:
                        state_possible_actions_dict[si] = {ai}
                    else:
                        state_possible_actions_dict[si].add(ai)

        # calculate value of the current state
        sorted_results = sorted(results)
        maxval, _, best_action = sorted_results[-1]
        value_this_state = 0.99*maxval + 0.01*sum([s[0] for s in results])/nr_actions # epsilon greedy policy with eps=0.01
        value_this_state *= 0.999 # discounting
        depth_searched = min([s[1] for s in results]) + 1

        # demo_states and connected states always get some value
        if value_this_state==0 and state.rambytes in state_possible_actions_dict:
            value_this_state = 0.1 * min(demo_values_thus_far)
        state_value_depth_dict[state.rambytes] = (value_this_state, depth_searched)

        # save best action for behavior cloning, if it is actually better than any other action
        minval = sorted_results[0][0]
        if maxval>minval:
            state_best_action_dict[state.rambytes] = best_action

        return value_this_state, depth_searched

    for stepnr in reversed(range(first_nr,len(demo_states)-args.frame_skip)):
        val, dep = search_states(demo_states[stepnr], args.search_depth)
        demo_values_thus_far.append(val)
        logger.log('nr %d working back %d steps, now at step %d. demo state has value %.4e, searched to depth %d' % (my_nr, len(demo_states), stepnr, val, dep))

    logger.log('nr %d processed demo of %d timesteps and extracted %d obs,action pairs' % (my_nr, len(demo_actions), len(state_best_action_dict)))

    # pickle the dicts
    extended_demo_file_name = os.path.join(logger.get_dir(), args.game + str(args.demo_nr) + '_extended_' + str(my_nr) + '.pkl')
    with open(extended_demo_file_name, "wb") as f:
        p = pickle.Pickler(f)
        p.fast = True
        p.dump({'state_best_action_dict': state_best_action_dict,
                'state_possible_actions_dict': state_possible_actions_dict})

def expand_demo_multi(startnr, stopnr, nr_workers):
    import multiprocessing
    for i in range(startnr, stopnr):
        p = multiprocessing.Process(target=expand_demo, args=(i, nr_workers))
        p.start()

if __name__ == '__main__':
    #call(expand_demo_multi, kwargs={'startnr':100, 'stopnr':116, 'nr_workers':10000}, log_relpath='atari-demo/test', num_cpu=16, num_gpu=0)

    nr_workers = 640
    for superworker in range(int(np.ceil(nr_workers/16))):
       startnr = superworker * 16
       stopnr = min([(superworker+1)*16, nr_workers])
       call(expand_demo_multi, kwargs={'startnr':startnr, 'stopnr':stopnr, 'nr_workers': nr_workers}, log_relpath='atari-demo/' + args.game + '/superworker'+str(superworker), num_cpu=16, num_gpu=0)

