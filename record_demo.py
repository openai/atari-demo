import os
import argparse
import numpy as np
import gym
from gym.envs.atari.atari_env import ACTION_MEANING
import pygame
from demo_wrapper import AtariDemo

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--game', type=str, default='MontezumaRevenge')
parser.add_argument('-f', '--frame_rate', type=int, default=60)
parser.add_argument('-y', '--screen_height', type=int, default=840)
parser.add_argument('-d', '--save_dir', type=str, default=None)
parser.add_argument('-n', '--demo_nr', type=int, default=0)
parser.add_argument('-s', '--frame_skip', type=int, default=4)
args = parser.parse_args()

if args.save_dir is None:
    save_dir = os.path.join(os.getcwd(), 'demos')
else:
    save_dir = args.save_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
demo_file_name = os.path.join(save_dir, args.game + str(args.demo_nr) + '.demo')


# //////// set up gym + atari part /////////
ACTION_KEYS = {
    "NOOP" : set(),
    "FIRE" : {'space'},
    "UP" : {'up'},
    "RIGHT": {'right'},
    "LEFT" : {'left'},
    "DOWN" : {'down'},
    "UPRIGHT" : {'up', 'right'},
    "UPLEFT" : {'up', 'left'},
    "DOWNRIGHT" : {'down', 'right'},
    "DOWNLEFT" : {'down', 'left'},
    "UPFIRE" : {'up', 'space'},
    "RIGHTFIRE" : {'right', 'space'},
    "LEFTFIRE" : {'left', 'space'},
    "DOWNFIRE" : {'down', 'space'},
    "UPRIGHTFIRE" : {'up', 'right', 'space'},
    "UPLEFTFIRE" : {'up', 'left', 'space'},
    "DOWNRIGHTFIRE" : {'down', 'right', 'space'},
    "DOWNLEFTFIRE" : {'down', 'left', 'space'},
    "TIMETRAVEL": {'b'}
}

env = AtariDemo(gym.make(args.game + 'NoFrameskip-v4'))
available_actions = [ACTION_MEANING[i] for i in env.unwrapped._action_set] + ["TIMETRAVEL"]
env.reset()
loaded_previous = False
if os.path.exists(demo_file_name):
    env.load_from_file(demo_file_name)
    loaded_previous = True

def get_gym_action(key_presses):
    action = 0
    for i,action_name in enumerate(available_actions):
        if ACTION_KEYS[action_name].issubset(key_presses):
            action = i
    return action


# ///////// set up pygame part //////////
pygame.init()
screen_size = (int((args.screen_height/210)*160),args.screen_height)
screen = pygame.display.set_mode(screen_size)
small_screen = pygame.transform.scale(screen.copy(), (160,210))
clock = pygame.time.Clock()
pygame.display.set_caption("Recording demonstration for " + args.game)

def show_text(text_lines):
    screen.fill((255, 255, 255))
    f1 = pygame.font.SysFont("", 30)
    for i, line in enumerate(text_lines):
        text = f1.render(line, True, (0, 0, 0))
        screen.blit(text, (50, 100 + 50 * i))
    pygame.display.flip()

def show_start_screen():
    text_lines = ["Recording demo for " + args.game,
                  "Control the game using the arrow keys and space bar",
                  "Hold <b> to go backward in time to fix mistakes",
                  "Press <s> to save the demo and exit",
                  "Press <SPACE BAR> to get started"]
    if loaded_previous:
        text_lines = text_lines[:1] + ["Continuing from previously recorded demo"] + text_lines[1:]
    show_text(text_lines)
    started = False
    while not started:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                key_name = pygame.key.name(event.key)
                if key_name == 'space':
                    started = True
        clock.tick(args.frame_rate)

def show_end_screen():
    text_lines = ["GAME OVER",
                  "Hold <b> to go backward in time",
                  "Press <s> to save the demo and exit"]
    show_text(text_lines)

def show_game_screen(observation):
    pygame.surfarray.blit_array(small_screen, np.transpose(observation,[1,0,2]))
    pygame.transform.scale(small_screen, screen_size, screen)
    pygame.display.flip()

key_is_pressed = set()
def process_key_presses():
    key_presses = set()
    quit = False
    save = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit = True
        elif event.type == pygame.KEYDOWN:
            key_name = pygame.key.name(event.key)
            key_presses.add(key_name)
            key_is_pressed.add(key_name)
        elif event.type == pygame.KEYUP:
            key_name = pygame.key.name(event.key)
            if key_name in key_is_pressed:
                key_is_pressed.remove(key_name)
            if key_name == 's':
                save = True
    key_presses.update(key_is_pressed)

    return key_presses, quit, save


# //////// run the game and record the demo! /////////
quit = False
done = False
show_start_screen()
while not quit:

    # process key presses & save when requested
    key_presses, quit, save = process_key_presses()
    if save:
        env.save_to_file(demo_file_name)
        quit = True

    # advance gym env
    action = get_gym_action(key_presses)
    for step in range(args.frame_skip):
        observation, reward, done, info = env.step(action)

    # show screen
    if done:
        show_end_screen()
    else:
        show_game_screen(observation)

    clock.tick(float(args.frame_rate)/args.frame_skip)
