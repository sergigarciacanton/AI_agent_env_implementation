# DEPENDENCIES
import copy
import sys
import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete
from typing import Tuple, Dict, Any, SupportsFloat, TypeVar, Union, List, Iterable
from Env_test.cav import CAV
from Env_test.config import (
    POSITION,
    TIMESTEPS_LIMIT,
    VECN_INIT,
    FECS_RANGE,
    BACKGROUND_VEHICLES,
    FEC_MAX_RAM,
    FEC_MAX_BW,
    FEC_MAX_GPU,
)
from Env_test.upc_graph import get_graph

ObsType = TypeVar("ObsType")


# FUNCTIONS
def find_fec(cav_trajectory: tuple) -> int:
    """
    Finds the key in the FECS_RANGE dictionary corresponding to the given tuple value.

    Parameters:
    - tuple_value (tuple): The tuple for which to find the key.

    Returns:
    - int or None: The key if found, otherwise None.

    Raises:
    - ValueError: If the tuple is not found in the dictionary.
    """
    try:
        for fec_id, trajectories in FECS_RANGE.items():
            if cav_trajectory in trajectories:
                return fec_id
        raise ValueError(f"Tuple {cav_trajectory} not found in the dictionary.")
    except ValueError as e:
        print(e)
        return -1


# CLASS
def check_range(value, min_val, max_val, label, key):
    if not (min_val <= value <= max_val):
        raise ValueError(
            f"\033[91mERROR: {label} out of range: {value}. Should be between {min_val} and {max_val} in FEC {key}.")


class Env_Test(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        # Attributes

        self.background_traffic = None
        self.reward = None
        self.terminated = None
        self.cav = None
        self.vecn = None
        self.timestep_limit = None

        # Action space
        self.action_space = Discrete(len(POSITION.keys()))

        # Observation space
        num_obs_features = 21  # Total number of observation features [VNF, CAV_info, VECN]
        self.observation_space = MultiDiscrete(np.array([1] * num_obs_features), dtype=np.int32)

        # Initialize Barcelona graph
        self.graph = get_graph()

    # Class methods

    def hops_to_target(self, cav: CAV) -> int:
        """
        Calculates the minimum number of hops from the current node to the destination node out of
        all possible shortest paths in the graph.

        Returns:
        int: Minimum number of hops from the current node to the destination node.
        """
        cav_destination = cav.vnf.get_request()[1]

        # Use networkx to find all shortest paths
        shortest_paths = nx.all_shortest_paths(self.graph, cav.current_n, cav_destination)

        # Calculate minimum hops from the shortest paths
        min_hops = min(len(path) - 1 for path in shortest_paths)

        return min_hops

    def initialize_cav(self, node_pairs) -> CAV:
        """
        Initialize and configure a CAV instance.

        Returns:
            CAV: The initialized CAV instance.
        """
        # Instantiate a CAV
        cav = CAV(node_pairs)

        # Set CAV id using its unique obj. id
        cav.set_cav_id(id(cav))

        # Set the calculated hops to the CAV
        cav.set_hops_to_target(self.hops_to_target(cav))

        return cav

    def new_cav(self, node_pairs=None) -> CAV:
        """
        Create a new CAV, ensuring resources in at least one closest FEC.

        Returns:
            CAV: The newly created CAV object.
        """
        max_iterations = 1000

        for iteration in range(1, max_iterations + 1):
            # Initialize a new CAV
            cav = self.initialize_cav(node_pairs)

            # Find the closest FECs to the current CAV's position
            potential_paths_and_fecs = [
                k
                for k, v in FECS_RANGE.items()
                for tuple_elem in v
                if tuple_elem[0] == cav.current_n
            ]

            # Check resources availability in the closest FECs
            fec_resources_status_dict = self.check_fec_resources(potential_paths_and_fecs, cav)

            # Exit the loop if there are resources in closest FECs
            if any(fec_resources_status_dict.values()):
                break
            else:
                # Delete the previous CAV and try again
                cav.delete_cav()

        else:
            # Raise an exception if no CAV is created after max_iterations
            raise RuntimeError(
                "Unable to create a CAV with available resources after {} iterations.".format(max_iterations))
            sys.exit(1)

        return cav

    def check_fec_resources(self, selected_fecs: Union[int, List[int], Iterable[int]], cav: CAV) -> Dict[int, bool]:

        # Extract VNF resource requirements from the CAV
        vehicle_vnf_resources = cav.vnf.get_request()[2:5]  # Extract only resource requirements

        # Dictionary to store the status of each FEC's resources
        fec_status: Dict[int, bool] = {}

        # If only a single FEC is provided, convert it to a list for uniform processing
        if isinstance(selected_fecs, int):
            selected_fecs = [selected_fecs]

        # Check resources in each selected FEC
        for fec in selected_fecs:
            # Check if resources in the FEC are sufficient for the CAV
            is_resources_sufficient = all(cav_vnf_request <= fec_resource for cav_vnf_request, fec_resource in
                                          zip(vehicle_vnf_resources, self.vecn[fec]))
            # Store the status in the dictionary
            fec_status[fec] = is_resources_sufficient

        return fec_status

    def _reward_fn(self, value):
        match value:
            case "yes_resources":
                # Calculate the cost of the trajectory from current node to the next node
                cav_trajectory_cost = -nx.path_weight(self.graph, [self.cav.previous_n, self.cav.current_n], 'weight')

                # Update the reward based on the trajectory cost
                self.reward += cav_trajectory_cost

                # Check if the current node has been revisited multiple times and add additional reward
                times_revisited_node = self.cav.cav_route.count(self.cav.current_n)
                self.reward += times_revisited_node * cav_trajectory_cost if times_revisited_node >= 2 else 0

                # Set the termination flag
                self.terminated = False

            case "no_resources":
                self.reward += -100
                # Set the termination flag
                self.terminated = True

            case "reached_destination":
                self.reward += 100

                # Get the start and end nodes from the request
                start_node, end_node, *_ = self.cav.vnf.get_request()

                # Find all possible shortest paths between the specified nodes
                all_possible_shortest_paths = list(nx.all_shortest_paths(self.graph, start_node, end_node, 'weight'))

                # Check if the current CAV route is among the shortest paths and update the reward accordingly
                self.reward += 200 if self.cav.cav_route in all_possible_shortest_paths else 0

                # Set the termination flag
                self.terminated = True

    def update_vehicle_info(self, action: Any, fec_to_request: int, vehicle: CAV) -> CAV:
        """
        Update information in the CAV object based on the given action and FEC identifier.

        Parameters:
        - action (Any): The action to update the current node and route in the CAV object.
        - fec_to_request (str): The FEC identifier to set in the CAV object.

        Raises:
        - ValueError: If the FEC identifier is not present in the dictionary.

        Returns:
        - None
        """
        # Update the current node and route in the CAV object
        vehicle.set_current_node(action)
        vehicle.add_node_to_route(action)

        # Check if FEC identifier is present in the dictionary
        if fec_to_request not in self.vecn:
            raise ValueError(f"FEC '{fec_to_request}' not found in the dictionary.")

        # Update FEC-related information in the CAV object
        vehicle.set_my_fec(fec_to_request)
        vehicle.set_hops_to_target(self.hops_to_target(vehicle))

        return vehicle

    def update_resources_in_previous_fec(self, vehicle: CAV) -> None:
        """
        Update resources in the previous FEC by adding back the resources allocated to the CAV's VNF.

        Raises:
        - ValueError: If the previous FEC identifier is not present in the dictionary.
        - ValueError: If the lengths of the previous FEC resources and allocated resources are not equal.

        Returns:
        - None
        """
        # Get previous FEC serving CAV's VNF
        previous_fec = vehicle.my_fec

        # CAV VNF
        cav_vnf_resources_req = vehicle.vnf.get_request()[2:5]

        # Check if previous FEC identifier is present in the dictionary
        if previous_fec not in self.vecn:
            raise ValueError(f"Previous FEC '{previous_fec}' not found in the dictionary.")

        # Retrieve previous FEC resources
        previous_fec_resources = self.vecn[previous_fec]

        # Check if lengths of previous FEC resources and allocated resources are equal
        if len(previous_fec_resources) != len(cav_vnf_resources_req):
            raise ValueError("Lengths of previous FEC resources and allocated resources must be equal.")

        # Update resources in the previous FEC by adding back the allocated resources
        updated_resources = tuple(x + y for x, y in zip(previous_fec_resources, cav_vnf_resources_req))

        # Update the previous FEC resources in the dictionary
        self.vecn[previous_fec] = updated_resources

    def update_selected_fec(self, fec_to_request: int, vehicle: CAV) -> None:
        """
        Update the resources in the selected FEC by subtracting the used resources.

        Parameters:
        - fec_to_request (int): The identifier of the FEC to be updated.

        Raises:
        - ValueError: If the FEC identifier is not present in the dictionary.
        - ValueError: If the lengths of the FEC resources and used resources are not equal.

        Returns:
        - None
        """
        # Check if FEC identifier is present in the dictionary
        if fec_to_request not in self.vecn:
            raise ValueError(f"FEC '{fec_to_request}' not found in the dictionary.")

        # Retrieve FEC resources
        fec_resources = self.vecn[fec_to_request]

        # CAV VNF
        used_resources = vehicle.vnf.get_request()[2:5]

        # Check if lengths of FEC resources and used resources are equal
        if len(fec_resources) != len(used_resources):
            raise ValueError("Lengths of FEC resources and used resources must be equal.")

        # Update resources in the selected FEC by subtracting the used resources
        updated_resources = tuple(x - y for x, y in zip(fec_resources, used_resources))

        # Update the FEC resources in the dictionary
        self.vecn[fec_to_request] = updated_resources

    def _get_obs(self):

        try:
            for key, values in self.vecn.items():
                gpu, ram, bw = values
                check_range(gpu, 0, FEC_MAX_GPU, "GPU", key)
                check_range(ram, 0, FEC_MAX_RAM, "RAM", key)
                check_range(bw, 0, FEC_MAX_BW, "Bandwidth", key)  # TODO revisar

        except ValueError as e:
            print(f"{e}")
            sys.exit(1)

        # VNF
        vnf = np.array(self.cav.vnf.get_request(), dtype=np.int32)
        # [source_n, target_n, gpu, ram, bw]

        # CAV info
        cav_info = np.array(self.cav.get_cav_info(self.timestep_limit), dtype=np.int32)
        # [current_n, my_fec, timestep_imit, hops_to_target]

        # VECN
        flatten_vecn_to_list = np.array([item for sublist in self.vecn.values() for item in sublist])
        # [fec0_gpu, fec0_ram, fec0_bw, ..., fec70_gpu, fec70_ram, fec70_bw]

        # Initial Obs
        observation = np.concatenate([vnf, cav_info, flatten_vecn_to_list], dtype=np.int32)
        # [VNF, CAV_info, VECN]

        return observation

    def move_background_traffic(self):
        """
        Move background traffic vehicles based on their trajectories and FEC.

        For each vehicle in the background traffic:
        1. Determine the appropriate FEC based on its trajectory.
        2. If the vehicle's current FEC differs from the selected FEC:
            a. Check if the resources in the requested FEC are available.
            b. Update the selected FEC for the vehicle and handle resource updates.
            c. If the vehicle is not newly instantiated, update resources in the previous FEC.
        3. Update the vehicle's information based on the new FEC and trajectory.
        4. If the vehicle has reached its final destination, update resources in the previous FEC.

        Returns:
            None
        """

        for vehicle_id, vehicle in self.background_traffic.items():
            # Select vehicle FEC
            vehicle_trajectory = (vehicle.prefix_route[0], vehicle.prefix_route[1])
            requesting_fec_for_vehicle = find_fec(vehicle_trajectory)

            # Vehicle FEC differs from selected FEC
            if vehicle.my_fec != requesting_fec_for_vehicle:

                if all(self.check_fec_resources(requesting_fec_for_vehicle, vehicle).values()):
                    self.update_selected_fec(requesting_fec_for_vehicle, vehicle)

                    # Only for not newly instantiated vehicles
                    if vehicle.my_fec != -1:
                        self.update_resources_in_previous_fec(vehicle)

                    # Update vehicle info
                    vehicle = self.update_vehicle_info(vehicle.prefix_route[1], requesting_fec_for_vehicle, vehicle)
                    vehicle.prefix_route.pop(0)

            elif vehicle.my_fec == requesting_fec_for_vehicle:
                vehicle = self.update_vehicle_info(vehicle.prefix_route[1], requesting_fec_for_vehicle, vehicle)
                vehicle.prefix_route.pop(0)

            # Check if the vehicle has reached its final destination
            if vehicle.current_n == vehicle.vnf.get_request()[1]:
                self.update_resources_in_previous_fec(vehicle)
                self.background_traffic[vehicle_id] = self.new_cav()

            # print(f"CAV: {self.cav.debug_cav(self.timestep_limit)}\n"
            #       f"VEH: {vehicle_id} - {vehicle.debug_cav(self.timestep_limit)}")

    def process_cav_trajectory(self, action: Any) -> None:
        """
        Process the trajectory of the Connected Autonomous Vehicle (CAV).

        Args:
            action: Description of the 'action' parameter.

        Returns:
            None
        """
        # Determine the next trajectory and FEC for the CAV
        cav_next_trajectory = (self.cav.current_n, action)
        fec_to_request = find_fec(cav_next_trajectory)
        fec_resources_status = all(self.check_fec_resources(fec_to_request, self.cav).values())

        # Check if CAV's FEC differs from the selected FEC
        if self.cav.my_fec != fec_to_request:
            # Handle the case when CAV's FEC differs from the selected FEC
            self._handle_different_fec(fec_to_request, fec_resources_status, action)
        else:
            # Handle the case when CAV stays within the same FEC coverage range
            self._handle_same_fec(action)

        # Check if CAV has reached its destination
        self._check_destination_reached()

    def _handle_different_fec(self, fec_to_request: Any, fec_resources_status: bool, action: Any) -> None:
        """
        Handle the case when CAV's FEC differs from the selected FEC.

        Args:
            fec_to_request: Description of the 'fec_to_request' parameter.
            fec_resources_status: Description of the 'fec_resources_status' parameter.
            action: Description of the 'action' parameter.

        Returns:
            None
        """
        # Check if there are sufficient resources in the selected FEC
        if fec_resources_status:
            # Update the selected FEC and resources
            self._update_selected_fec_and_resources(action, fec_to_request)
            # Update CAV information based on the action and selected FEC
            self.cav = self.update_vehicle_info(action, fec_to_request, self.cav)
            # Provide a reward for successfully obtaining resources
            self._reward_fn("yes_resources")
        else:
            # Provide a reward for unsuccessful attempt to obtain resources
            self._reward_fn("no_resources")

    def _handle_same_fec(self, action: Any) -> None:
        """
        Handle the case when CAV stays within the same FEC coverage range.

        Args:
            action: Description of the 'action' parameter.

        Returns:
            None
        """
        # Update CAV information based on the action and existing FEC
        self.cav = self.update_vehicle_info(action, self.cav.my_fec, self.cav)
        # Provide a reward for staying within the same FEC coverage range
        self._reward_fn("yes_resources")

    def _check_destination_reached(self) -> None:
        """
        Check if the CAV has reached its destination.

        Returns:
            None
        """
        # Check if the current position of the CAV matches its destination
        if self.cav.current_n == self.cav.vnf.get_request()[1]:
            # Update resources in the previous FEC and provide a reward for reaching the destination
            self.update_resources_in_previous_fec(self.cav)
            self._reward_fn("reached_destination")

    def _update_selected_fec_and_resources(self, action: Any, fec_to_request: Any) -> None:
        """
        Update the selected FEC and resources when the CAV's FEC differs from the selected FEC.

        Args:
            action: Description of the 'action' parameter.
            fec_to_request: Description of the 'fec_to_request' parameter.

        Returns:
            None
        """
        # Update the selected FEC
        self.update_selected_fec(fec_to_request, self.cav)
        # Check if the CAV is not newly instantiated
        if self.cav.my_fec != -1:
            # Update resources in the previous FEC
            self.update_resources_in_previous_fec(self.cav)

    # *************************************** RESET ********************************************************************
    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> Tuple[Any, Dict[str, Any]]:
        super().reset(seed=seed)

        # Timesteps limit count initialization
        self.timestep_limit = TIMESTEPS_LIMIT

        # Initialize Vehicular Edge Computing Network
        self.vecn = copy.deepcopy(VECN_INIT)

        # Initialize the CAV
        self.cav = self.new_cav()

        # Initial Obs
        initial_obs = self._get_obs()

        # Info
        info = {}

        self.reward = 0

        # Initialize background vehicles, each with its VNF and prefix route
        self.background_traffic = {i: self.new_cav() for i in range(BACKGROUND_VEHICLES)}
        # print(f"\nRESET\n{self.cav.debug_cav(self.timestep_limit)}")
        return initial_obs, info

    # *************************************** STEP *********************************************************************
    def step(self, action: Any) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # Update timestep limit counter
        self.timestep_limit -= 1

        # print("\nSTEP")

        # Move CAV
        self.process_cav_trajectory(action)

        # Move background traffic vehicles
        self.move_background_traffic()

        # Check for timesteps limit truncation
        truncated = self.timestep_limit <= 0

        # Info
        if self.terminated and self.reward >= 100:
            info = {'count': 1}
        else:
            info = {'count': 0}

        # Obs
        observation = self._get_obs()
        # if self.terminated: print(f"done: {self.reward}")
        # print(self.vecn)
        return observation, self.reward, self.terminated, truncated, info
