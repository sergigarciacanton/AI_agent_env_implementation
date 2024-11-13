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
from graph_upc import get_graph
from typing import Optional, Dict, Any, Tuple
from config import (
    TIMESTEPS_LIMIT,
    FECS_RANGE,
    NODES_POSITION,
    BACKGROUND_VEHICLES,
    BACKGROUND_VEHICLES_ROUTE_NODES

)
from itertools import chain
from gymnasium.spaces import Discrete, MultiDiscrete
from background_vehicles import BACKGROUND_VEHICLE
import zmq


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
def new_background_vehicle(node_pairs_for_bg_vehicles):
    # Instantiate new vehicle
    vehicle = BACKGROUND_VEHICLE(nodes_for_bg_vehicles=node_pairs_for_bg_vehicles)
    # Set vehicle id
    vehicle.set_vehicle_id(id(vehicle))

    return vehicle


class EnvironmentUPC(gym.Env):
    # Initialize any parameters or variables needed for the environment
    def __init__(self,):
        self.vecn = None
        self.used_fec_resources = None
        self.timesteps_limit = None
        self.background_vehicles = None
        self.reward = None
        self.graph = get_graph()
        self.fec_dict = dict()
        self.vnf_and_cav_info = dict()
        self.terminated = False
        self.cav = None
        self.cav_route = []
        self.state_changed = False
        config = configparser.ConfigParser()

        config.read("/home/user/Documents/AI_agent_env_implementation/ini_files/env_annex.ini")
        self.general = config['general']

        self.logger = logging.getLogger('env')
        self.logger.setLevel(int(self.general['log_level']))
        self.logger.addHandler(logging.FileHandler(self.general['log_file_name'], mode='w', encoding='utf-8'))
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(ColoredFormatter('%(log_color)s%(message)s'))
        self.logger.addHandler(stream_handler)
        logging.getLogger('pika').setLevel(logging.WARNING)
        self.context = zmq.Context()
        self.zeromq_subscribe_thread = threading.Thread(target=self.subscribe)
        self.cav_thread = None

        # Observation space
        num_obs_features = 14  # Total number of observation features [VNF, CAV_info, VECN]
        self.observation_space = MultiDiscrete(np.array([1] * num_obs_features), dtype=np.int32)

        # Action space
        self.action_space = Discrete(len(NODES_POSITION.keys()))

        #Set up ZeroMQ
        host = self.general['control_ip']
        zero_port = self.general['zeromq_port']
        self.zero_conn = self.context.socket(zmq.SUB)
        address = "tcp://" + host + ":" + zero_port
        self.zero_conn.connect(address)
        self.zero_conn.subscribe("")
        self.zeromq_subscribe_thread.daemon = True
        self.zeromq_subscribe_thread.start()

    def check_fec_resources(self, fec_id):
        return (
                self.vnf_and_cav_info[1]['gpu'] <= self.fec_dict[fec_id]['gpu'] and
                self.vnf_and_cav_info[1]['ram'] <= self.fec_dict[fec_id]['ram'] and
                self.vnf_and_cav_info[1]['bw'] <= self.fec_dict[fec_id]['bw']
        )

    def _reward_fn(self, cav_next_node) -> None:
        """
        Calculate the reward for the CAV based on its last trajectory.

        Returns: Tuple[float, bool]: A tuple containing the reward value and a boolean indicating whether the CAV
        completed the route.
        """
        self.terminated = False

        # CAV data
        vnf_source_node = self.vnf_and_cav_info[1]['source']
        vnf_target_node = self.vnf_and_cav_info[1]['target']
        cav_current_node = self.vnf_and_cav_info[1]['current_node']

        # All possible shortest paths from source_node to target_node
        all_possible_shortest_paths = list(
            nx.all_shortest_paths(self.graph, vnf_source_node, vnf_target_node, 'weight'))

        # Calculate the negative reward based on the path weight
        if self.cav_route.count(cav_next_node) >= 2:
            times_revisited_node = self.cav_route.count(cav_next_node)
            self.reward += times_revisited_node * (
                -nx.path_weight(self.graph, [cav_current_node, cav_next_node], 'weight'))

        else:
            self.reward += -nx.path_weight(self.graph, [cav_current_node, cav_next_node], 'weight')

        # Check if CAV's route is completed
        if vnf_target_node == cav_next_node:
            self.terminated = True
            self.reward += 100
            # Check if completed CAV's route is one of the shortest paths
            if self.cav_route in all_possible_shortest_paths:
                self.reward += 200

    def subscribe(self):
        stop = False
        self.logger.info('[I] Waiting for published data...')
        message = None
        while not stop:
            try:
                message = json.loads(self.zero_conn.recv_json())
            except zmq.ContextTerminated:
                pass

            self.logger.debug("[D] Received message. Key: " + str(message["key"]) + ". Message: " + message["body"])

            if str(message["key"]) == 'fec':
                self.fec_dict = {int(k): v for k, v in json.loads(message["body"]).items()}
            elif str(message["key"]) == 'vnf':
                self.vnf_and_cav_info = {int(k): v for k, v in json.loads(message["body"]).items()}
                self.state_changed = True

    def start_cav(self,):
        self.cav = CAV(self.nodes_to_evaluate)

    def get_obs(self):
        vnf = copy.deepcopy(self.vnf_and_cav_info[1])
        vnf.pop('previous_node')

        fec_copy = copy.deepcopy(self.fec_dict)
        fecs = list()
        for fec in fec_copy.values():
            fec.pop('ip')
            fec.pop('connected_users')
            fecs.append(list(fec.values()))

        obs = np.array([
            list(vnf.values()) +
            [self.timesteps_limit] +
            list(chain.from_iterable(fecs))
        ],
            dtype=np.int16)[0]

        return obs

    def send_action_to_fec(self, action, fec_id):
        host = self.fec_dict[fec_id]['ip']
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

    def hops_to_target(self, current_node, target_node):
        # Hops to target node
        hops = []
        for path in nx.all_shortest_paths(self.graph,
                                          current_node,
                                          target_node,
                                          weight='weight'):
            # Save the number of hops from each path
            hops.append(len(path) - 1)

        return min(hops)

    def close(self):
        self.context.term()
        killed_threads = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(self.subscribe_thread.ident),
                                                                    ctypes.py_object(SystemExit))
        if killed_threads == 0:
            raise ValueError("Thread ID " + str(self.zeromq_subscribe_thread.ident) + " does not exist!")
        elif killed_threads > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(self.zeromq_subscribe_thread.ident, 0)
        self.logger.debug('[D] Successfully killed thread ' + str(self.zeromq_subscribe_thread.ident))

    def process_cav_trajectory(self, action):
        # Determine the next trajectory and FEC for the CAV
        current_node = int(self.vnf_and_cav_info[1]['current_node'])
        next_node = action
        next_cav_trajectory = (current_node, next_node)
        fec_to_request = get_next_hop_fec(next_cav_trajectory)

        # Check if CAV's FEC differs from the selected FEC
        current_fec_connection = self.vnf_and_cav_info[1]['cav_fec']
        if fec_to_request != current_fec_connection:
            fec_resource_ok = self.check_fec_resources(fec_to_request)
        else:
            fec_resource_ok = True

        if fec_resource_ok:  # Resources OK
            self.cav_route.append(action)
            self._reward_fn(action)
            self.send_action_to_fec(action, self.vnf_and_cav_info[1]['cav_fec'])

            # Wait for a state change
            while not self.state_changed:
                time.sleep(0.001)
            self.state_changed = False

        elif not fec_resource_ok:  # Resources not OK
            self.reward -= 100
            self.terminated = True
            self.logger.warning('[!] Not enough available resources!')

    def check_episode_ending(self, truncated, vnf_and_cav_info_copy):
        # CAV reaches destination
        if 1 not in self.vnf_and_cav_info.keys():
            # Update CAV info
            vnf_and_cav_info_copy['previous_node'] = vnf_and_cav_info_copy['current_node']
            vnf_and_cav_info_copy['current_node'] = vnf_and_cav_info_copy['target']

            # Remove obsolete info
            vnf_and_cav_info_copy.pop('previous_node')
            # vnf_and_cav_info_copy.pop('time_steps')

            # Get VECN
            fec_copy = copy.deepcopy(self.fec_dict)
            fecs = list()
            for fec in fec_copy.values():
                fec.pop('connected_users')
                fec.pop('ip')
                fecs.append(list(fec.values()))

            # Obs
            next_obs = \
                np.array([
                    list(vnf_and_cav_info_copy.values()) +
                    [self.timesteps_limit] +
                    list(chain.from_iterable(fecs))
                ],
                    dtype=np.int16)[0]

            self.terminated = True

            if self.general['training_if'] == 'y' or self.general['training_if'] == 'Y':
                # Kill CAV thread
                self.cav_thread.join()

        # CAV still moves
        else:
            # Obs
            next_obs = self.get_obs()
            # manage truncation among FECs
            if truncated:
                self.send_action_to_fec(-1, self.vnf_and_cav_info[1]['cav_fec'])
                while not self.state_changed:
                    time.sleep(0.001)
                self.state_changed = False
                if 1 in self.vnf_and_cav_info.keys():
                    self.logger.error('[!] Truncated VNF not killed!')
                else:
                    time.sleep(0.003)  # Just to give time to FECs to remove VNF from their lists

        return next_obs

    def update_vecn_status(self):
        for fec_id, fec_resources in self.fec_dict.items():
            self.vecn[fec_id]['gpu'] = fec_resources['gpu'] - self.used_fec_resources[fec_id][0]
            self.vecn[fec_id]['ram'] = fec_resources['ram'] - self.used_fec_resources[fec_id][1]
            self.vecn[fec_id]['bw'] = fec_resources['bw'] - self.used_fec_resources[fec_id][2]

    def move_background_vehicles(self):
        # Iterate over all background vehicles
        for vehicle_id, vehicle in self.background_vehicles.items():
            # Next potential vehicle trajectory
            vehicle_trajectory = (vehicle.prefix_route[0], vehicle.prefix_route[1])
            # Potential next FEC
            requesting_fec_for_vehicle = get_next_hop_fec(vehicle_trajectory)
            # Vehicle VNF resource request
            vnf_gpu, vnf_ram, vnf_bw = vehicle.get_vnf_resources_request()

            # Vehicle FEC differs from selected FEC
            if vehicle.my_fec != requesting_fec_for_vehicle:

                # Check FEC resource availability
                if self.vecn[requesting_fec_for_vehicle]['gpu'] >= vnf_gpu and \
                        self.vecn[requesting_fec_for_vehicle]['ram'] >= vnf_ram and \
                        self.vecn[requesting_fec_for_vehicle]['bw'] >= vnf_bw:

                    # Update FEC resources
                    self.used_fec_resources[requesting_fec_for_vehicle][0] += vnf_gpu
                    self.used_fec_resources[requesting_fec_for_vehicle][1] += vnf_ram
                    self.used_fec_resources[requesting_fec_for_vehicle][2] += vnf_bw

                    # Only for not newly instantiated vehicles
                    if vehicle.my_fec != -1:
                        self.used_fec_resources[vehicle.my_fec][0] -= vnf_gpu
                        self.used_fec_resources[vehicle.my_fec][1] -= vnf_ram
                        self.used_fec_resources[vehicle.my_fec][2] -= vnf_bw

                    # Update VECN status
                    self.update_vecn_status()

                    # Update vehicle FEC
                    vehicle.set_my_fec(requesting_fec_for_vehicle)

                    # Move vehicle
                    vehicle.set_current_node(vehicle.prefix_route[1])
                    vehicle.prefix_route.pop(0)

            elif vehicle.my_fec == requesting_fec_for_vehicle:
                # Move vehicle
                vehicle.set_current_node(vehicle.prefix_route[1])
                vehicle.prefix_route.pop(0)

            # Check if the vehicle has reached its final destination
            if vehicle.current_n == vehicle.vnf.get_request()['target']:
                # Free resources on last connected FEC
                self.used_fec_resources[vehicle.my_fec][0] -= vnf_gpu
                self.used_fec_resources[vehicle.my_fec][1] -= vnf_ram
                self.used_fec_resources[vehicle.my_fec][2] -= vnf_bw
                # Instantiate a new
                self.background_vehicles[vehicle_id] = new_background_vehicle(BACKGROUND_VEHICLES_ROUTE_NODES)

    # *************************************** RESET ********************************************************************
    def reset(self, seed: int = None, nodes_to_evaluate=None) -> Tuple[Any, Dict[str, Any]]:
        super().reset(seed=seed)

        # Timesteps count initialization
        self.timesteps_limit = TIMESTEPS_LIMIT

        self.nodes_to_evaluate = nodes_to_evaluate

        if self.general['training_if'] == 'y' or self.general['training_if'] == 'Y':
            # CAV initialization
            start = time.time()
            self.cav_thread = threading.Thread(target=self.start_cav)
            self.cav_thread.start()
            end = time.time() - start
            # print(f"[DEBUG] Init CAV time: {end}")

        # Logs
        self.logger.debug('Starting new episode...')

        # Waiting for a state change
        start = time.time()
        while not self.state_changed:
            time.sleep(0.001)
        self.state_changed = False
        end = time.time() -start
        # print(f"[DEBUG] State changed time: {end}")

        # Build CAV route
        self.cav_route = []
        self.cav_route.append(self.vnf_and_cav_info[1]['current_node'])

        # Initial obs
        initial_obs = self.get_obs()

        # Info
        info = {}

        # Initial reward
        self.reward = 0

        # Initialize background vehicles, each with its VNF and prefix route
        self.background_vehicles = {i: new_background_vehicle(BACKGROUND_VEHICLES_ROUTE_NODES) for i in
                                    range(BACKGROUND_VEHICLES)}

        # Vehicular Edge Computing Network (vecn) status only for background vehicles
        # fec_id : (gpu, ram, bw)
        self.used_fec_resources = {i: [0, 0, 0] for i in FECS_RANGE.keys()}

        # Copy of VECN status from control
        self.vecn = copy.deepcopy(self.fec_dict)

        return initial_obs, info

    # *************************************** STEP *********************************************************************
    def step(self, action):

        # Timestep limit count
        self.timesteps_limit -= 1

        # Copy VNF and CAV info from control
        vnf_and_cav_info_copy = copy.deepcopy(self.vnf_and_cav_info[1])

        # Move CAV
        self.process_cav_trajectory(action)

        # Check episode ending due to CAV reaching target node or episode truncation
        truncated = self.timesteps_limit <= 0
        next_obs = self.check_episode_ending(truncated, vnf_and_cav_info_copy)

        # Move background traffic
        self.move_background_vehicles()

        if self.terminated and self.reward >= 100:
            info = {'count': 1}
        else:
            info = {'count': 0}

        # Logs
        self.logger.debug('[D] Sending information to agent. obs = ' + str(next_obs) +
                          ', reward = ' + str(self.reward) +
                          ',terminated = ' + str(self.terminated) +
                          ', truncated  = ' + str(truncated))

        # Update final next_obs
        vecn = [
            list(self.vecn[0].values())[1:4],
            list(self.vecn[1].values())[1:4]
        ]

        next_obs[8:] = np.array(vecn).flatten()

        return next_obs, self.reward, self.terminated, truncated, info


