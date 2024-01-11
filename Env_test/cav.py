"""Author: Carlos Ruiz de Mendoza Date: 21/11/2023 Description: This script defines a CAV (Connected and Autonomous
Vehicle) class with various methods for managing its state."""

# DEPENDENCIES
from Env_test.vnf_generator import VNF
from Env_test.upc_graph import get_graph
import networkx as nx


# CLASSES
class CAV:
    def __init__(self, nodes_for_bg_vehicles, nodes_to_evaluate,  initial_node=None, initial_id=None, ):
        """
        Initialize a CAV instance with default values.

        Parameters:
        - initial_node (optional): The initial node for the CAV.
        - initial_id (optional): The initial identifier for the CAV.
        """
        self.id = initial_id  # Obj id
        self.vnf = VNF(nodes_for_bg_vehicles, nodes_to_evaluate)  # Random VNF
        self.current_n = initial_node if initial_node is not None else self.vnf.get_request()[0]
        self.previous_n = self.current_n
        self.cav_route = [self.current_n]
        self.my_fec = -1
        # self.hops_to_target = 0
        self.prefix_route = nx.dijkstra_path(G=get_graph(), source=self.vnf.get_request()[0],
                                             target=self.vnf.get_request()[1])  # Only for background vehicles

    def set_cav_id(self, cav_id):
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

    def add_node_to_route(self, node):
        """
        Add each node that the CAV visits until route completion.

        Parameters:
        - node (int): The node to add to the CAV route.
        """
        # Check for duplicates before adding the node.
        self.cav_route.append(node)

    def set_my_fec(self, mec_id):
        """
        Update the MEC id to which the CAV is connected.

        Parameters:
        - mec_id (int): The identifier of the MEC node.
        """
        # Validate mec_id as a valid MEC node.
        self.my_fec = mec_id

    def set_hops_to_target(self, hops):
        self.hops_to_target = hops

    def get_cav_info(self, remain_timesteps):
        """
        Return CAV information including current node, associated MEC, and remaining timesteps.

        Parameters:
        - remain_timesteps (int): The remaining timesteps in the current operation.

        Returns:
        A list [int, int, int]: Current node, MEC identifier, and remaining timesteps.
        """
        # Provide context on the meaning of remaining timesteps.
        # return [self.current_n, self.my_fec, remain_timesteps, self.hops_to_target]
        return [self.current_n, self.my_fec, remain_timesteps]

    def delete_cav(self):
        """
        Delete the CAV instance and its VNF.

        Warning: This action cannot be undone and frees up resources.
        """
        # Provide a warning or confirmation mechanism before deletion.
        del self.vnf
        del self

    def debug_cav(self, timestep_limit):
        """
        Return debug information about the CAV instance.

        Returns:
        A tuple (str, int, int, list, int): Debug information about the CAV instance.
        """
        # Include additional debugging information.
        return {'id':             self.id,
                'current_n':      self.current_n,
                'previous_n':     self.previous_n,
                'My FEC':         self.my_fec,
                'source_n':       self.vnf.get_request()[0],
                'target_n':       self.vnf.get_request()[1],
                'gpu':            self.vnf.get_request()[2],
                'ram':            self.vnf.get_request()[3],
                'bw':             self.vnf.get_request()[4],
                # 'Timestep limit': timestep_limit,
                # 'prefix_route_BG_vehicles': self.prefix_route,
                # 'route':          self.cav_route,
}