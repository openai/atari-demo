import pickle
import gym
from gym import spaces

class AtariDemo(gym.Wrapper):
    """
        Records actions taken, creates checkpoints, allows time travel, restoring and saving of states
    """

    def __init__(self, env):
        super(AtariDemo, self).__init__(env)
        self.action_space = spaces.Discrete(len(env.unwrapped._action_set)+1) # add "time travel" action
        self.save_every_k = 100
        self.max_time_travel_steps = 10000

    def _step(self, action):
        if action >= len(self.env.unwrapped._action_set):
            obs, reward, done, info = self.time_travel()

        else:
            if self.steps_in_the_past > 0:
                self.restore_past_state()

            if len(self.done)>0 and self.done[-1]:
                obs = self.obs[-1]
                reward = 0
                done = True
                info = None

            else:
                obs, reward, done, info = self.env.step(action)

                self.actions.append(action)

                self.obs.append(obs)
                self.reward.append(reward)
                self.done.append(done)
                self.info.append(info)
                if len(self.obs) > self.max_time_travel_steps:
                    self.obs.pop(0)
                    self.reward.pop(0)
                    self.done.pop(0)
                    self.info.pop(0)

            # periodic checkpoint saving
            if not done:
                if (len(self.checkpoint_action_nr)>0 and len(self.actions) >= self.checkpoint_action_nr[-1] + self.save_every_k) \
                        or (len(self.checkpoint_action_nr)==0 and len(self.actions) >= self.save_every_k):
                    self.save_checkpoint()

        return obs, reward, done, info

    def _reset(self):
        self.actions = []
        self.checkpoints = []
        self.checkpoint_action_nr = []
        self.obs = []
        self.reward = []
        self.done = []
        self.info = []
        self.steps_in_the_past = 0
        return self.env.reset()

    def time_travel(self):
        if len(self.obs) > 1:
            reward = self.reward.pop()
            self.obs.pop()
            self.done.pop()
            self.info.pop()
            obs = self.obs[-1]
            done = self.done[-1]
            info = self.info[-1]
            self.steps_in_the_past += 1

        else: # reached time travel limit
            reward = 0
            obs = self.obs[0]
            done = self.done[0]
            info = self.info[0]

        # rewards are differences in subsequent state values, and so should get reversed sign when going backward in time
        reward = -reward

        return obs, reward, done, info

    def save_to_file(self, file_name):
        dat = {'actions': self.actions, 'checkpoints': self.checkpoints, 'checkpoint_action_nr': self.checkpoint_action_nr}
        with open(file_name, "wb") as f:
            pickle.dump(dat, f)

    def load_from_file(self, file_name):
        self._reset()
        with open(file_name, "rb") as f:
            dat = pickle.load(f)
        self.actions = dat['actions']
        self.checkpoints = dat['checkpoints']
        self.checkpoint_action_nr = dat['checkpoint_action_nr']
        self.load_state_and_walk_forward()

    def save_checkpoint(self):
        chk_pnt = self.env.unwrapped.clone_state()
        self.checkpoints.append(chk_pnt)
        self.checkpoint_action_nr.append(len(self.actions))

    def restore_past_state(self):
        self.actions = self.actions[:-self.steps_in_the_past]
        while len(self.checkpoints)>0 and self.checkpoint_action_nr[-1]>len(self.actions):
            self.checkpoints.pop()
            self.checkpoint_action_nr.pop()
        self.load_state_and_walk_forward()
        self.steps_in_the_past = 0

    def load_state_and_walk_forward(self):
        if len(self.checkpoints)==0:
            self.env.reset()
            time_step = 0
        else:
            self.env.unwrapped.restore_state(self.checkpoints[-1])
            time_step = self.checkpoint_action_nr[-1]

        for a in self.actions[time_step:]:
            action = self.env.unwrapped._action_set[a]
            self.env.unwrapped.ale.act(action)

