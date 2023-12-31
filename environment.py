# DEPENDENCIES
import configparser
import logging
import sys
import threading
import time
import pika
import json
import ctypes
import socket
import numpy as np
import networkx as nx
import copy
import gymnasium as gym
from colorlog import ColoredFormatter
from CAV import CAV
from Utils.graph_upc import get_graph
from typing import Optional
from config import (
    TIMESTEPS_LIMIT,
    FECS_RANGE,
    NODES_POSITION,
)
from itertools import chain
from gymnasium.spaces import Discrete, MultiDiscrete

# GLOBALS
FEC_MIN_GPU = 0
FEC_MIN_RAM = 0
FEC_MIN_BW = 0
FEC_MAX_GPU = 128
FEC_MAX_RAM = 32
FEC_MAX_BW = 100


# FUNCTIONS
def get_next_hop_fec(cav_trajectory) -> Optional[int]:
    """
    Retrieves the FEC associated with the last hop in the given CAV trajectory.

    Parameters:
    - cav_trajectory (Tuple[int, int]): The CAV trajectory represented as a tuple of current and previous nodes.

    Returns: - Optional[int]: The FEC (Forward Error Correction) associated with the last hop, or None if no matching
    FEC is found.
    """

    try:
        return next((fec for fec, one_hop_path in FECS_RANGE.items() if cav_trajectory in one_hop_path))
    except StopIteration:
        raise ValueError("No matching FEC found for the given CAV trajectory.")


# CLASSES
class EnvironmentUPC(gym.Env):
    # Initialize any parameters or variables needed for the environment
    def __init__(self):
        self.graph = get_graph()
        self.fec_list = dict()
        self.vnf_list = dict()
        self.timesteps_limit = 20
        self.terminated = False
        self.cav = None
        self.cav_route = []
        self.state_changed = False
        config = configparser.ConfigParser()

        config.read("/home/upc_ai_vecn/Documents/AI_agent_env_implementation/ini_files/env_annex.ini")
        self.general = config['general']

        self.logger = logging.getLogger('env')
        self.logger.setLevel(int(self.general['log_level']))
        self.logger.addHandler(logging.FileHandler(self.general['log_file_name'], mode='w', encoding='utf-8'))
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(ColoredFormatter('%(log_color)s%(message)s'))
        self.logger.addHandler(stream_handler)
        logging.getLogger('pika').setLevel(logging.WARNING)
        self.rabbit_conn = pika.BlockingConnection(
            pika.ConnectionParameters(host=self.general['control_ip'], port=self.general['rabbit_port'],
                                      credentials=pika.PlainCredentials(self.general['control_username'],
                                                                        self.general['control_password'])))
        self.subscribe_thread = threading.Thread(target=self.subscribe, args=(self.rabbit_conn, 'fec vnf'))
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()
        self.cav_thread = None

        # Observation space
        # self.observation_space = Tuple((
        #     # VNF
        #     Box(min(NODES_POSITION.keys()), max(NODES_POSITION.keys()), shape=(1,), dtype=int),  # Starting CAV node
        #     Box(min(NODES_POSITION.keys()), max(NODES_POSITION.keys()), shape=(1,), dtype=int),  # Destination CAV node
        #     Box(0, 18, shape=(1,), dtype=int),  # GPU CAV request
        #     Box(0, 18, shape=(1,), dtype=int),  # RAM CAV request
        #     Box(0, 18, shape=(1,), dtype=int),  # BW CAV request
        #     # CAV info
        #     Box(min(NODES_POSITION.keys()), max(NODES_POSITION.keys()), shape=(1,), dtype=int),  # Current CAV node
        #     Box(min(FECS_RANGE.keys()), max(FECS_RANGE.keys()), shape=(1,), dtype=int),  # Current CAV FEC connection
        #     Box(0, TIMESTEPS_LIMIT, shape=(1,), dtype=int),  # Remain CAV times-steps
        #     Box(0, 100, shape=(1,), dtype=int),  # Hops to target node
        #     # VECN status
        #     Box(FEC_MIN_GPU, FEC_MAX_GPU, shape=(1,), dtype=int),  # FEC0 free GPU
        #     Box(FEC_MIN_RAM, FEC_MAX_RAM, shape=(1,), dtype=int),  # FEC0 free RAM
        #     Box(FEC_MIN_BW, FEC_MAX_BW, shape=(1,), dtype=int),  # FEC0 free BW
        #     Box(FEC_MIN_GPU, FEC_MAX_GPU, shape=(1,), dtype=int),  # FEC1 free GPU
        #     Box(FEC_MIN_RAM, FEC_MAX_RAM, shape=(1,), dtype=int),  # FEC1 free RAM
        #     Box(FEC_MIN_BW, FEC_MAX_BW, shape=(1,), dtype=int),  # FEC1 free BW
        #     Box(FEC_MIN_GPU, FEC_MAX_GPU, shape=(1,), dtype=int),  # FEC2 free GPU
        #     Box(FEC_MIN_RAM, FEC_MAX_RAM, shape=(1,), dtype=int),  # FEC2 free RAM
        #     Box(FEC_MIN_BW, FEC_MAX_BW, shape=(1,), dtype=int),  # FEC2 free BW
        #     Box(FEC_MIN_GPU, FEC_MAX_GPU, shape=(1,), dtype=int),  # FEC3 free GPU
        #     Box(FEC_MIN_RAM, FEC_MAX_RAM, shape=(1,), dtype=int),  # FEC3 free RAM
        #     Box(FEC_MIN_BW, FEC_MAX_BW, shape=(1,), dtype=int))  # FEC3 free BW
        # )

        # Observation space
        num_obs_features = 21  # Total number of observation features [VNF, CAV_info, VECN]
        self.observation_space = MultiDiscrete(np.array([1] * num_obs_features), dtype=np.int32)

        # Action space
        self.action_space = Discrete(len(NODES_POSITION.keys()))

    def check_fec_resources(self, fec_id):
        return (
                self.vnf_list['1']['gpu'] <= self.fec_list[str(fec_id)]['gpu'] and
                self.vnf_list['1']['ram'] <= self.fec_list[str(fec_id)]['ram'] and
                self.vnf_list['1']['bw'] <= self.fec_list[str(fec_id)]['bw']
        )

    def _reward_fn(self, cav_next_node) -> None:
        """
        Calculate the reward for the CAV based on its last trajectory.

        Returns: Tuple[float, bool]: A tuple containing the reward value and a boolean indicating whether the CAV
        completed the route.
        """
        self.terminated = False

        # CAV data
        vnf_source_node = self.vnf_list['1']['source']
        vnf_target_node = self.vnf_list['1']['target']
        cav_current_node = self.vnf_list['1']['current_node']

        # All possible shortest paths from source_node to target_node
        all_possible_shortest_paths = list(
            nx.all_shortest_paths(self.graph, vnf_source_node, vnf_target_node, 'weight'))

        # Calculate the negative reward based on the path weight
        if self.cav_route.count(cav_next_node) >= 2:
            times_revisited_node = self.cav_route.count(cav_next_node)
            self.reward += times_revisited_node * (-nx.path_weight(self.graph, [cav_current_node, cav_next_node], 'weight'))

        else:
            self.reward += -nx.path_weight(self.graph, [cav_current_node, cav_next_node], 'weight')

        # Check if CAV's route is completed
        if vnf_target_node == cav_next_node:
            self.terminated = True
            self.reward += 100
            # Check if completed CAV's route is one of the shortest paths
            if self.cav_route in all_possible_shortest_paths:
                self.reward += 100

    def subscribe(self, conn, key_string):
        channel = conn.channel()

        channel.exchange_declare(exchange=self.general['control_exchange_name'], exchange_type='direct')

        queue = channel.queue_declare(queue='', exclusive=True).method.queue

        keys = key_string.split(' ')
        for key in keys:
            channel.queue_bind(
                exchange=self.general['control_exchange_name'], queue=queue, routing_key=key)

        self.logger.info('[I] Waiting for published data...')

        def callback(ch, method, properties, body):
            self.logger.debug("[D] Received. Key: " + str(method.routing_key) + ". Message: " + body.decode("utf-8"))
            if str(method.routing_key) == 'fec':
                self.fec_list = json.loads(body.decode('utf-8'))
            elif str(method.routing_key) == 'vnf':
                self.vnf_list = json.loads(body.decode('utf-8'))
                self.state_changed = True

        channel.basic_consume(
            queue=queue, on_message_callback=callback, auto_ack=True)

        channel.start_consuming()

    def start_cav(self):
        self.cav = CAV()

    def get_obs(self):
        vnf = copy.deepcopy(self.vnf_list['1'])
        vnf.pop('previous_node')
        vnf.pop('time_steps')
        hops = []
        for path in nx.all_shortest_paths(self.graph, self.vnf_list["1"]['current_node'], self.vnf_list["1"]['target'],
                                          weight='weight'):
            # Save the number of hops from each path
            hops.append(len(path) - 1)

        hops_to_target = [min(hops)]
        fec_copy = copy.deepcopy(self.fec_list)
        fecs = list()
        for fec in fec_copy.values():
            fec.pop('ip')
            fec.pop('connected_users')
            fecs.append(list(fec.values()))

        obs = np.array([list(vnf.values()) + [self.timesteps_limit] + hops_to_target + list(chain.from_iterable(fecs))],
                       dtype=np.int16)[0]
        return obs

    def send_action_to_fec(self, action, fec_id):
        host = self.fec_list[str(fec_id)]['ip']
        port = int(self.general['agent_fec_port'])
        fec_socket = socket.socket()
        fec_socket.connect((host, port))
        fec_socket.send(json.dumps(dict(type="action", action=action)).encode())
        response = json.loads(fec_socket.recv(1024).decode())
        if response['res'] == 200:
            self.logger.debug('[D] Action ' + str(action) + ' sent successfully to FEC ' + str(fec_id))
        else:
            self.logger.critical('[!] Error from FEC' + str(response['res']))
            raise Exception
        fec_socket.close()

    def reset(self, seed=None):
        # Reset the environment to its initial state
        # Return the initial observation
        super().reset(seed=seed)

        # CAV initialization
        self.cav_thread = threading.Thread(target=self.start_cav)
        self.cav_thread.start()

        # Timesteps count initialization
        self.timesteps_limit = TIMESTEPS_LIMIT

        self.logger.debug('Starting new episode...')

        while not self.state_changed:
            time.sleep(0.001)

        self.state_changed = False
        self.cav_route = []
        self.cav_route.append(self.vnf_list['1']['current_node'])

        # Initial obs
        initial_obs = self.get_obs()

        # Info
        info = {}

        self.reward = -1

        return initial_obs, info

    def step(self, action):
        # Take an action in the environment
        # Update the state, provide a reward, and check for termination
        # Return the next observation, reward, termination flag, and additional information
        self.timesteps_limit -= 1

        next_cav_trajectory = (int(self.vnf_list['1']['current_node']), int(action))
        fec_id = get_next_hop_fec(next_cav_trajectory)

        if fec_id != self.vnf_list['1']['cav_fec']:
            fec_resource_ok = self.check_fec_resources(fec_id)
        else:
            fec_resource_ok = True
        old_cav_state = copy.deepcopy(self.vnf_list['1'])

        if fec_resource_ok:  # Resources OK
            self.cav_route.append(action)
            self._reward_fn(action)
            self.send_action_to_fec(action, self.vnf_list['1']['cav_fec'])

            while not self.state_changed:
                time.sleep(0.001)
            self.state_changed = False

        elif not fec_resource_ok:  # Resources not OK
            self.reward -= 100
            self.terminated = True

        truncated = self.timesteps_limit <= 0

        if '1' not in self.vnf_list.keys():
            old_cav_state.pop('previous_node')
            old_cav_state.pop('time_steps')
            hops = []
            for path in nx.all_shortest_paths(self.graph, old_cav_state['current_node'],
                                              old_cav_state['target'],
                                              weight='weight'):
                # Save the number of hops from each path
                hops.append(len(path) - 1)

            hops_to_target = [min(hops)]
            fec_copy = copy.deepcopy(self.fec_list)
            fecs = list()
            for fec in fec_copy.values():
                fec.pop('ip')
                fec.pop('connected_users')
                fecs.append(list(fec.values()))

            next_obs = \
                np.array([list(old_cav_state.values()) + [self.timesteps_limit] + hops_to_target + list(
                    chain.from_iterable(fecs))],
                         dtype=np.int16)[0]
            self.terminated = True
            self.cav_thread.join()
        else:
            next_obs = self.get_obs()

            if truncated:
                self.send_action_to_fec(-1, self.vnf_list['1']['cav_fec'])
                while not self.state_changed:
                    time.sleep(0.001)
                self.state_changed = False
                if '1' in self.vnf_list.keys():
                    self.logger.error('[!] Truncated VNF not killed!')
                else:
                    time.sleep(0.003)  # Just to give time FECs to remove VNF from their lists

        if self.terminated and self.reward >= 100:
            info = {'count': 1}
        else:
            info = {'count': 0}

        self.logger.debug('[D] Sending information to agent. obs = ' + str(next_obs) + ', reward = ' + str(self.reward)
                          + ',terminated = ' + str(self.terminated) + ', truncated  = ' + str(truncated))

        return next_obs, self.reward, self.terminated, truncated, info

    def close(self):
        killed_threads = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(self.subscribe_thread.ident),
                                                                    ctypes.py_object(SystemExit))
        if killed_threads == 0:
            raise ValueError("Thread ID " + str(self.subscribe_thread.ident) + " does not exist!")
        elif killed_threads > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(self.subscribe_thread.ident, 0)
        self.logger.debug('[D] Successfully killed thread ' + str(self.subscribe_thread.ident))
        self.subscribe_thread.join()
