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
from Env_test.config_test import (

    SEED,
    NODES_2_TRAIN,
    VNF_GPU,
    VNF_RAM,
    VNF_BW
)

from Env_test.upc_graph import get_graph
from typing import List, Tuple, Optional

# GLOBALS
UPC_GRAPH = get_graph()

# Activate deterministic randomness
if SEED is not None:
    random.seed(SEED)


# FUNCTIONS
def set_nodes_for_route(nodes_for_bg_vehicles, nodes_to_evaluate):
    # Nodes only for background vehicle
    if nodes_for_bg_vehicles is not None:
        source_n, target_n = random.sample(nodes_for_bg_vehicles, 2)
    # Nodes for evaluation
    elif nodes_to_evaluate is not None:
        source_n, target_n = nodes_to_evaluate
    # Random nodes for training
    else:
        source_n, target_n = random.sample(NODES_2_TRAIN, 2)
    return source_n, target_n


# CLASS
class VNF:
    """ Generate random VNF requests"""

    def __init__(self, nodes_for_bg_vehicles, nodes_to_evaluate):
        # Set the seed
        self.source_n, self.target_n = set_nodes_for_route(nodes_for_bg_vehicles, nodes_to_evaluate)
        self.gpu = random.choice(VNF_GPU)
        self.ram = random.choice(VNF_RAM)
        self.bw = random.choice(VNF_BW)

    def get_request(self):
        return [self.source_n, self.target_n, self.gpu, self.ram, self.bw]

    def set_vnf(self, ns, nt):
        self.source_n = ns
        self.target_n = nt

