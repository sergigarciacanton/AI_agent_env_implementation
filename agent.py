from environment import Environment
import configparser
from colorlog import ColoredFormatter
import logging
import sys
import time


def get_action(target, curr_node):
    if curr_node == 0:
        return 4
    elif curr_node == 1:
        return 0
    elif curr_node == 2:
        if target == 0 or target == 1 or target == 4 or target == 5:
            return 1
        else:
            return 6
    elif curr_node == 3:
        return 2
    elif curr_node == 4:
        if target == 8 or target == 9 or target == 12 or target == 13:
            return 8
        else:
            return 5
    elif curr_node == 5:
        if target == 0 or target == 1 or target == 4 or target == 8 or target == 12 or target == 13:
            return 1
        else:
            return 6
    elif curr_node == 6:
        if target == 0 or target == 1 or target == 2 or target == 3 or target == 4 or target == 7:
            return 7
        else:
            return 10
    elif curr_node == 7:
        return 3
    elif curr_node == 8:
        return 12
    elif curr_node == 9:
        if target == 8 or target == 11 or target == 12 or target == 13 or target == 14 or target == 15:
            return 8
        else:
            return 5
    elif curr_node == 10:
        if target == 2 or target == 3 or target == 7 or target == 11 or target == 14 or target == 15:
            return 14
        else:
            return 9
    elif curr_node == 11:
        if target == 2 or target == 3 or target == 6 or target == 7:
            return 7
        else:
            return 10
    elif curr_node == 12:
        return 13
    elif curr_node == 13:
        if target == 10 or target == 11 or target == 14 or target == 15:
            return 14
        else:
            return 9
    elif curr_node == 14:
        return 15
    elif curr_node == 15:
        return 11


config = configparser.ConfigParser()
config.read("agent_outdoor.ini")
general = config['general']
locations = config['general']

logger = logging.getLogger('')
logger.setLevel(int(general['log_level']))
logger.addHandler(logging.FileHandler(general['log_file_name'], mode='w', encoding='utf-8'))
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(ColoredFormatter('%(log_color)s%(message)s'))
logger.addHandler(stream_handler)
logging.getLogger('pika').setLevel(logging.WARNING)
env = Environment(logger, general)
try:
    num_episodes = 100
    before = time.time()
    for episode in range(num_episodes):
        logger.info('[I] Starting new episode')
        obs, info = env.reset()
        episode_reward = 0
        logger.info('[I] Got new observation: ' + str(obs))

        for step in range(25):
            # action = select_action_strategy_fn(obs)
            action = get_action(obs[1], obs[5])
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                logger.info('[I] Episode terminated')
                break
            else:
                logger.info('[I] Got new observation: ' + str(obs))
    env.close()
    after = time.time()
    diff = after - before

    logger.info('[I] Time elapsed for ' + str(num_episodes) + ' episodes: ' + str(diff) + ' s')
except KeyboardInterrupt:
    env.close()
