from collections import deque
from PIL import Image
import numpy as np
import pickle
import gym
from gym import spaces

'''
Bunch of gym wrappers. Some of these are near copies of wrappers in other repos, but we change them slightly such that reset() can return an info dict.
'''

class AtariDemo(gym.Wrapper):
    """
        Records actions taken, creates checkpoints, allows time travel, restoring and saving of states
    """

    def __init__(self, env, disable_time_travel=False):
        super(AtariDemo, self).__init__(env)
        self.action_space = spaces.Discrete(len(env.unwrapped._action_set)+1) # add "time travel" action
        self.save_every_k = 100
        self.max_time_travel_steps = 10000
        self.disable_time_travel = disable_time_travel

    def _step(self, action):
        if action >= len(self.env.unwrapped._action_set):
            if self.disable_time_travel:
                obs, reward, done, info = self.env.step(0)
            else:
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
        obs = self.env.reset()
        self.actions = []
        self.checkpoints = []
        self.checkpoint_action_nr = []
        self.obs = [obs]
        self.reward = [0]
        self.done = [False]
        self.info = [None]
        self.steps_in_the_past = 0
        return obs

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

class ReplayResetEnv(gym.Wrapper):
    """
        Randomly resets to states from a replay
    """

    def __init__(self, env, demo_file_name, seed, reset_prob=0.001, reset_steps_ignored=400):
        super(ReplayResetEnv, self).__init__(env)
        with open(demo_file_name, "rb") as f:
            dat = pickle.load(f)
        self.actions = dat['actions']
        self.checkpoints = dat['checkpoints']
        self.checkpoint_action_nr = dat['checkpoint_action_nr']
        self.rng = np.random.RandomState(seed)
        self.reset_prob = reset_prob
        self.reset_steps_ignored = reset_steps_ignored
        self.actions_to_overwrite = []
        self.next_reset_from_replay = False

    def _step(self, action):
        if len(self.actions_to_overwrite) > 0:
            action = self.actions_to_overwrite.pop(0)
            valid = False
        else:
            valid = True
        obs, reward, done, info = self.env.step(action)

        if valid and self.rng.rand() < self.reset_prob:
            self.next_reset_from_replay = True
            done = True
            info['replay_reset.random_reset'] = True
            valid = False
        if not valid:
            info['replay_reset.invalid_transition'] = True

        return obs, reward, done, info

    def _reset(self):
        if self.next_reset_from_replay:
            action_nr_to_reset_to = self.rng.choice(len(self.actions)+1)
            start_action_nr = 0
            start_ckpt = None
            for nr, ckpt in zip(self.checkpoint_action_nr[::-1], self.checkpoints[::-1]):
                if nr <= (action_nr_to_reset_to - self.reset_steps_ignored):
                    start_action_nr = nr
                    start_ckpt = ckpt
                    break
            if start_action_nr > 0:
                self.env.unwrapped.restore_state(start_ckpt)
            nr_to_start_lstm = np.maximum(action_nr_to_reset_to-self.reset_steps_ignored, start_action_nr)
            for a in self.actions[start_action_nr:nr_to_start_lstm]:
                action = self.env.unwrapped._action_set[a]
                self.env.unwrapped.ale.act(action)
            self.actions_to_overwrite = self.actions[nr_to_start_lstm:action_nr_to_reset_to]
            self.next_reset_from_replay = False
            obs = self.env.unwrapped._get_image()
        else:
            obs = self.env.reset()

        return obs

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.was_real_done  = True

    def _step(self, action):
        prev_lives = self.env.unwrapped.ale.lives()
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < prev_lives and lives > 0:
            # for Qbert somtimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        return obs, reward, done, info

    def _reset(self):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset()
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def _step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        combined_info = {}
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if info is not None:
                combined_info.update(info)
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, combined_info

    def _reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs,info = obs
            self._obs_buffer.append(obs)
            return obs, info
        else:
            self._obs_buffer.append(obs)
            return obs

class FastSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)
        self._skip = skip

    def _step(self, action):
        reward = 0.0
        done = False
        for _ in range(self._skip):
            reward += self.env.unwrapped.ale.act(self.env.unwrapped._action_set[action])
            if self.env.unwrapped.ale.game_over():
                done = True
                break

        return None, reward, done, {}

class EnvState(object):
    def __init__(self, ale_state, ram, obs, game_over):
        self.ale_state = ale_state
        self.ram = ram
        self.rambytes = self.ram.tostring()
        self.obs = obs
        self.game_over = game_over

class RamState(object):
    def __init__(self, ram):
        self.ram = ram
        self.state_hash = hash(self.ram.tostring())

    def __hash__(self):
        return self.state_hash

    def __eq__(self, other):
        return np.all(self.ram == other.ram)

class ResetWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.obs = None
        self.ram = None
        self.game_over = None
        self.times_cloned = None

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.obs = obs
        self.ram = self.env.unwrapped._get_ram()
        self.game_over = done
        return obs, reward, done, info

    def _reset(self):
        obs = self.env.reset()
        self.times_cloned = 0
        self.obs = obs
        self.ram = self.env.unwrapped._get_ram()
        self.game_over = False
        return obs

    def clone_state(self):
        self.times_cloned += 1
        ale_state = self.env.unwrapped.clone_state()
        state = EnvState(ale_state, self.ram, self.obs, self.game_over)
        return state

    def restore_state(self, state):
        if self.times_cloned > 1e4: # bug workaround
            self._reset()
        self.obs = state.obs
        self.ram = state.ram
        self.game_over = state.game_over
        self.env.unwrapped.restore_state(state.ale_state)

class EpsilonGreedyEnv(gym.Wrapper):
    def __init__(self, env, eps=0.01):
        gym.Wrapper.__init__(self, env)
        self.eps = eps

    def _step(self, action):
        if np.random.rand()<self.eps:
            action = np.random.randint(self.env.action_space.n)
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

class ClipRewardEnv(gym.RewardWrapper):
    def _reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)

class StickyActionEnv(gym.Wrapper):
    def __init__(self, env, stick_prob=0.25):
        gym.Wrapper.__init__(self, env)
        self.stick_prob = stick_prob
        self.last_action = 0

    def _step(self, action):
        if np.random.rand() < self.stick_prob:
            action = self.last_action
        self.last_action = action
        return self.env.step(action)

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.res = 84
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.res, self.res, 1))

    def _observation(self, obs):
        has_info = False
        if isinstance(obs, tuple):
            obs,info = obs
            has_info = True
        frame = np.dot(obs.astype('float32'), np.array([0.299, 0.587, 0.114], 'float32'))
        frame = np.array(Image.fromarray(frame).resize((self.res, self.res),
            resample=Image.BILINEAR), dtype=np.uint8)
        frame = frame.reshape((self.res, self.res, 1))
        if has_info:
            return frame, info
        else:
            return frame

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def _reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=8):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        if isinstance(env.action_space, gym.spaces.MultiBinary):
            # used for retro environments
            self.noop_action = np.zeros(self.env.action_space.n, dtype=np.int64)
        else:
            # used for atari environments
            self.noop_action = 0
            assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def _reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

class VideoWriter(gym.Wrapper):
    def __init__(self, env, video_writer):
        gym.Wrapper.__init__(self, env)
        self.video_writer = video_writer

    def process_frame(self, frame):
        f_out = np.zeros((224, 160, 3), dtype=np.uint8)
        f_out[7:-7, :] = np.cast[np.uint8](frame)
        return f_out

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.video_writer.append_data(self.process_frame(obs))
        return obs, reward, done, info

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Buffer observations and stack across channels (last axis)."""
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        assert shp[2] == 1  # can only stack 1-channel frames
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], k))

    def _reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        ob = self.env.reset()
        if isinstance(ob, tuple):
            ob,info = ob
            has_info = True
        else:
            has_info = False
        for _ in range(self.k): self.frames.append(ob)
        if has_info:
            return self._observation(), info
        else:
            return self._observation()

    def _step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._observation(), reward, done, info

    def _observation(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=2)

def wrap_replay_reset(env, demo_file_name, seed, episode_life=True, clip_rewards=True):
    assert 'NoFrameskip' in env.spec.id  # required for DeepMind-style skip
    env = ReplayResetEnv(env, demo_file_name, seed)
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'Pong' in env.spec.id:
        env = FireResetEnv(env)
    env = MaxAndSkipEnv(env, skip=4)
    env = WarpFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    return env

def wrap_deepmind(env, episode_life=False, clip_rewards=False, frame_stack=True, noop_reset=False, sticky_actions=False, epsilon_greedy=False):
    assert 'NoFrameskip' in env.spec.id
    if episode_life:
        env = EpisodicLifeEnv(env)
    if noop_reset:
        env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = WarpFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    if sticky_actions:
        env = StickyActionEnv(env)
    if epsilon_greedy:
        env = EpsilonGreedyEnv(env)
    return env

def wrap_deepmind_no_frameskip(env, episode_life=False, clip_rewards=False, frame_stack=False, noop_reset=False, sticky_actions=False, epsilon_greedy=False):
    assert 'NoFrameskip' in env.spec.id
    if episode_life:
        env = EpisodicLifeEnv(env)
    if noop_reset:
        env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=1)
    env = WarpFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    if sticky_actions:
        env = StickyActionEnv(env)
    if epsilon_greedy:
        env = EpsilonGreedyEnv(env)
    return env

