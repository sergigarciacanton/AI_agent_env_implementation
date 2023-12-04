from environment import Environment
import configparser
from colorlog import ColoredFormatter
import logging
import sys


def get_action(target, curr_node):
    if curr_node == 1:
        if target != 5 and target != 9 and target != 13:
            return 2
        else:
            return 5
    elif curr_node == 2:
        if target == 6 or target == 10 or target == 14:
            return 6
        elif target == 1 or target == 5 or target == 9 or target == 13:
            return 1
        else:
            return 3
    elif curr_node == 3:
        if target == 7 or target == 11 or target == 15:
            return 7
        elif target == 4 or target == 8 or target == 12 or target == 16:
            return 4
        else:
            return 2
    elif curr_node == 4:
        if target != 8 and target != 12 and target != 16:
            return 3
        else:
            return 8
    elif curr_node == 5:
        if target == 1:
            return 1
        elif target == 9 or target == 13:
            return 9
        else:
            return 6
    elif curr_node == 6:
        if target == 2 or target == 3:
            return 2
        elif target == 1 or target == 5 or target == 9:
            return 5
        elif target == 4 or target == 7 or target == 8 or target == 11 or target == 12:
            return 7
        else:
            return 10
    elif curr_node == 7:
        if target == 2 or target == 3:
            return 3
        elif target == 4 or target == 8 or target == 12:
            return 8
        elif target == 1 or target == 5 or target == 6 or target == 9 or target == 10:
            return 6
        else:
            return 11
    elif curr_node == 8:
        if target == 4:
            return 4
        elif target == 12 or target == 16:
            return 12
        else:
            return 7
    elif curr_node == 9:
        if target == 13:
            return 13
        elif target == 1 or target == 5:
            return 5
        else:
            return 10
    elif curr_node == 10:
        if target == 14 or target == 15:
            return 14
        elif target == 13 or target == 5 or target == 9:
            return 9
        elif target == 16 or target == 7 or target == 8 or target == 11 or target == 12:
            return 11
        else:
            return 6
    elif curr_node == 11:
        if target == 14 or target == 15:
            return 15
        elif target == 16 or target == 8 or target == 12:
            return 12
        elif target == 13 or target == 5 or target == 6 or target == 9 or target == 10:
            return 10
        else:
            return 7
    elif curr_node == 12:
        if target == 16:
            return 16
        elif target == 4 or target == 8:
            return 8
        else:
            return 11
    elif curr_node == 13:
        if target != 5 and target != 9 and target != 1:
            return 14
        else:
            return 9
    elif curr_node == 14:
        if target == 6 or target == 10 or target == 2:
            return 10
        elif target == 1 or target == 5 or target == 9 or target == 13:
            return 13
        else:
            return 15
    elif curr_node == 15:
        if target == 7 or target == 11 or target == 3:
            return 11
        elif target == 4 or target == 8 or target == 12 or target == 16:
            return 16
        else:
            return 14
    elif curr_node == 16:
        if target != 8 and target != 12 and target != 4:
            return 15
        else:
            return 12


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

    for episode in range(100):
        logger.info('[I] Starting new episode')
        obs, info = env.reset()
        episode_reward = 0
        logger.info('[I] Got new observation: ' + str(obs))

        for step in range(25):
            # action = select_action_strategy_fn(obs)
            action = get_action(obs['vnf']['target'], obs['vnf']['current_node'])
            obs, reward, terminated, info = env.step(action, obs['vnf']['cav_fec'])
            if terminated:
                logger.info('[I] Episode terminated')
                break
            else:
                logger.info('[I] Got new observation: ' + str(obs))
    env.stop()
except KeyboardInterrupt:
    env.stop()
