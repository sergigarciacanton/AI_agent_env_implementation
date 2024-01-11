"""Author: Carlos Ruiz de Mendoza Date: 21/11/2023 Description: This script defines a CAV (Connected and Autonomous
Vehicle) class with various methods for managing its state."""

# DEPENDENCIES
from Utils.vnf_generator import VNF
from Utils.graph_upc import get_graph
import networkx as nx


# CLASSES
class BACKGROUND_VEHICLE:
    def __init__(self, nodes_for_bg_vehicles=None, initial_id=None):
        self.id = initial_id  # Obj id
        self.vnf = VNF(nodes_for_bg_vehicles)  # Random VNF
        self.current_n = self.vnf.get_request()['source']
        self.previous_n = self.current_n
        self.my_fec = -1
        self.prefix_route = nx.dijkstra_path(
            G=get_graph(),
            source=self.vnf.get_request()['source'],
            target=self.vnf.get_request()['target']
        )

    def get_vnf_resources_request(self):
        return self.vnf.get_request()['gpu'], self.vnf.get_request()['ram'], self.vnf.get_request()['bw']

    def set_vehicle_id(self, cav_id):
        """
        Set the CAV object id.

        Parameters:
        - cav_id (int): The identifier to set for the CAV.
        """
        # Validate cav_id as a non-negative integer.
        self.id = cav_id

    def set_current_node(self, node):
        """
        Update the current CAV node.

        Parameters:
        - node (int): The new node for the CAV.
        """
        # Validate that the provided node is a valid network node.
        self.previous_n = self.current_n
        self.current_n = node

    def set_my_fec(self, mec_id):
        """
        Update the MEC id to which the CAV is connected.

        Parameters:
        - mec_id (int): The identifier of the MEC node.
        """
        # Validate mec_id as a valid MEC node.
        self.my_fec = mec_id

    def get_background_vehicle_info(self, remain_timesteps):
        """
        Return background vehicle information including current node, associated MEC, and remaining timesteps.

        Parameters:
        - remain_timesteps (int): The remaining timesteps in the current operation.

        Returns:
        A list [int, int, int]: Current node, MEC identifier, and remaining timesteps.
        """
        # Provide context on the meaning of remaining timesteps.
        # return [self.current_n, self.my_fec, remain_timesteps, self.hops_to_target]
        return [self.current_n, self.my_fec, remain_timesteps]

    def delete_vehicle(self):
        """
        Delete the CAV instance and its VNF.

        Warning: This action cannot be undone and frees up resources.
        """
        # Provide a warning or confirmation mechanism before deletion.
        del self.vnf
        del self

    def debug_background_vehicle(self, timestep_limit):
        """
        Return debug information about the CAV instance.

        Returns:
        A tuple (str, int, int, list, int): Debug information about the CAV instance.
        """
        # Include additional debugging information.
        return {'id': self.id,
                'current_n': self.current_n,
                'previous_n': self.previous_n,
                'My FEC': self.my_fec,
                'source_n': self.vnf.get_request()['source'],
                'target_n': self.vnf.get_request()['target'],
                'gpu': self.vnf.get_request()['gpu'],
                'ram': self.vnf.get_request()['ram'],
                'bw': self.vnf.get_request()['bw'],
                'prefix_route_BG_vehicles': self.prefix_route,
                }

