"""
Author: Carlos Ruiz de Mendoza
Date: 21/11/2023
Description: Generates VNF requests; [source_node, target_node, gpu_req, ram_req, bw_req]. 'source_node'
and 'target_node', are according to the possible routes within the Barcelona scenario represented as a weighted
directed graph.
"""

# DEPENDENCIES
import random
import networkx as nx
from Env_test.config import (
    MAX_GPU,
    MIN_GPU,
    MAX_RAM,
    MIN_RAM,
    MAX_BW,
    MIN_BW,
    SEED,
    NODES_2_TRAIN,
)

from Env_test.upc_graph import get_graph
from typing import List, Tuple, Optional

# GLOBALS
UPC_GRAPH = get_graph()

# Activate deterministic randomness
if SEED is not None:
    random.seed(SEED)


# FUNCTIONS
def get_source_and_target_nodes(node_pairs: Optional[List[Tuple]] = None) -> Tuple[int, int]:
    """
    Get source and target nodes for training or evaluation.

    Parameters:
    - node_pairs (Optional[List[Tuple]]): List of node pairs for evaluation.

    Returns:
    - Tuple[int, int]: Source and target nodes.
    """
    # All possible nodes to train on
    nodes = NODES_2_TRAIN

    if node_pairs is not None:  # Only if evaluating the model
        source_node, target_node = node_pairs[0], node_pairs[1]
        return source_node, target_node
    else:  # Training mode
        while True:
            # Randomly select two nodes
            source_node, target_node = 3, 2 #random.sample(nodes, 2)

            # Check if a path exists between selected nodes
            if nx.has_path(UPC_GRAPH, source_node, target_node):
                return source_node, target_node


# CLASS
class VNF:
    """ Generate random VNF requests"""

    def __init__(self, node_pairs=None):
        # Set the seed
        self.source_n, self.target_n = get_source_and_target_nodes(node_pairs)
        self.gpu = random.choice([MIN_GPU, MAX_GPU])
        self.ram = random.choice([MIN_RAM, MAX_RAM])
        self.bw = random.choice([MIN_BW, MAX_BW])

    def get_request(self):
        return [self.source_n, self.target_n, self.gpu, self.ram, self.bw]

    def set_vnf(self, ns, nt):
        self.source_n = ns
        self.target_n = nt
