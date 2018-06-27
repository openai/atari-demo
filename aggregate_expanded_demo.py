import argparse
import pickle
import numpy as np
from rcall import gce
import os.path as osp
from atari_demo.utils import save_as_pickled_object

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--game', type=str, default='MontezumaRevenge')
parser.add_argument('-n', '--demo_nr', type=int, default=0)
args = parser.parse_args()

state_best_action_dict = {}
state_possible_actions_dict = {}

nr_workers = 640
for superworker in range(int(np.ceil(nr_workers/16))):
    gce.main('pull', '--rel=atari-demo/' + args.game + '/superworker' + str(superworker), '--delete')
    startnr = superworker * 16
    stopnr = min([(superworker+1)*16, nr_workers])
    print('superworker %d' % superworker)
    for my_nr in range(startnr, stopnr):
        print(my_nr)
        fname = osp.join(osp.expanduser('~'),'data/rcall/gce/atari-demo/' + args.game + '/superworker' + str(superworker) + '/' + args.game + str(args.demo_nr) + '_extended_' + str(my_nr) + '.pkl')
        with open(fname, 'rb') as f:
            dat = pickle.load(f)
            state_best_action_dict.update(dat['state_best_action_dict'])
            state_possible_actions_dict.update(dat['state_possible_actions_dict'])

print('extracted %d states and %d best actions' % (len(state_possible_actions_dict), len(state_best_action_dict)))

# if data was gathered using older version of code, recompute hashes
ks = [s for s in state_best_action_dict.keys()]
if not isinstance(ks[0], bytes):
    state_possible_actions_dict = {s.ram.tostring():a for s,a in state_possible_actions_dict.items()}
    state_best_action_dict = {s.ram.tostring():a for s,a in state_best_action_dict.items()}

fpath = osp.join(osp.expanduser('~'),'data/rcall/gce/atari-demo/' + args.game + str(args.demo_nr) + '_extended.pkl')
dat = {'best_action_dict': state_best_action_dict, 'possible_actions_dict': state_possible_actions_dict}
save_as_pickled_object(dat, fpath)

