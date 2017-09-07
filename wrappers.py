import pickle
import numpy as np
import gym
from collections import deque
from PIL import Image
from gym import spaces

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
        self._skip       = skip

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
        self._obs_buffer.append(obs)
        return obs

class ClipRewardEnv(gym.RewardWrapper):
    def _reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.res = 84
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.res, self.res, 1))

    def _observation(self, obs):
        frame = np.dot(obs.astype('float32'), np.array([0.299, 0.587, 0.114], 'float32'))
        frame = np.array(Image.fromarray(frame).resize((self.res, self.res),
            resample=Image.BILINEAR), dtype=np.uint8)
        return frame.reshape((self.res, self.res, 1))

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
        for _ in range(self.k): self.frames.append(ob)
        return self._observation()

    def _step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._observation(), reward, done, info

    def _observation(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=2)

def wrap_deepmind(env, episode_life=True, clip_rewards=True):
    """Configure environment for DeepMind-style Atari.
    Note: this does not include frame stacking!"""
    assert 'NoFrameskip' in env.spec.id  # required for DeepMind-style skip
    if episode_life:
        env = EpisodicLifeEnv(env)
    env = MaxAndSkipEnv(env, skip=4)
    if 'Pong' in env.spec.id:
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    env = FrameStack(env, 4)
    return env

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

def wrap_deepmind_npz(env, episode_life=False, clip_rewards=False):
    """Configure environment for DeepMind-style Atari.
    Note: this does not include frame stacking!"""
    assert 'NoFrameskip' in env.spec.id  # required for DeepMind-style skip
    if episode_life:
        env = EpisodicLifeEnv(env)
    env = MaxAndSkipEnv(env, skip=4)
    if 'Pong' in env.spec.id:
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    env = FrameStack(env, 4)
    return env