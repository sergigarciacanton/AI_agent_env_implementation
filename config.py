# # ******************************* ENVIRONMENT *******************************
#
# NODES_2_TRAIN = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# BACKGROUND_VEHICLES_ROUTE_NODES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# TIMESTEPS_LIMIT: int = 20
# MODEL_PATH: str = "/home/user/Documents/AI_agent_env_implementation/Agents/Models/Rainbow/"
# SERGI_PLOTS: str = "/home/user/Documents/AI_agent_env_implementation/Sergi_plots/All_nodes_8_bg_vehicles/"
# SEED: int = 1976
# BACKGROUND_VEHICLES = 10
# # ******************************* GRAPH **************************************
# # FECs coverage range
# FECS_RANGE = {
#     0: [(1, 0), (0, 4), (4, 5), (5, 1), (2, 1), (9, 5)],
#     1: [(3, 2), (2, 6), (6, 7), (7, 3), (5, 6), (11, 7)],
#     2: [(8, 12), (12, 13), (13, 9), (9, 8), (4, 8), (10, 9)],
#     3: [(15, 11), (11, 10), (10, 14), (14, 15), (13, 14), (6, 10)]
# }
#
# # Manually specify scenario nodes positions for Manhattan layout
# NODES_POSITION = {
#     0: (0, 3), 1: (1, 3), 2: (2, 3), 3: (3, 3),
#     4: (0, 2), 5: (1, 2), 6: (2, 2), 7: (3, 2),
#     8: (0, 1), 9: (1, 1), 10: (2, 1), 11: (3, 1),
#     12: (0, 0), 13: (1, 0), 14: (2, 0), 15: (3, 0),
# }
#
# # Scenario edge cost
# EDGES_COST = {
#     (0, 4): {'weight': 0.1},
#     (4, 8): {'weight': 0.1},
#     (8, 12): {'weight': 0.1},
#     (13, 9): {'weight': 0.5},
#     (9, 5): {'weight': 0.5},
#     (5, 1): {'weight': 0.5},
#     (2, 6): {'weight': 0.5},
#     (6, 10): {'weight': 0.5},
#     (10, 14): {'weight': 0.5},
#     (15, 11): {'weight': 0.1},
#     (11, 7): {'weight': 0.1},
#     (7, 3): {'weight': 0.1},
#     (3, 2): {'weight': 0.1},
#     (2, 1): {'weight': 0.1},
#     (1, 0): {'weight': 0.1},
#     (4, 5): {'weight': 0.5},
#     (5, 6): {'weight': 0.5},
#     (6, 7): {'weight': 0.5},
#     (11, 10): {'weight': 0.5},
#     (10, 9): {'weight': 0.5},
#     (9, 8): {'weight': 0.5},
#     (12, 13): {'weight': 0.1},
#     (13, 14): {'weight': 0.1},
#     (14, 15): {'weight': 0.1},
# }
#
# # ***************************** VNF *******************************************
#
# VNF_GPU: list = [2, 4]
# VNF_RAM: list = [4, 8]
# VNF_BW: list = [5, 25]
#
# FEC_MAX_GPU: int = 16
# FEC_MAX_RAM: int = 32
# FEC_MAX_BW: int = 100
#
# VECN_INIT: dict = {
#     0: (FEC_MAX_GPU, FEC_MAX_RAM, FEC_MAX_BW),
#     1: (FEC_MAX_GPU, FEC_MAX_RAM, FEC_MAX_BW),
#     2: (FEC_MAX_GPU, FEC_MAX_RAM, FEC_MAX_BW),
#     3: (FEC_MAX_GPU, FEC_MAX_RAM, FEC_MAX_BW),
# }
#
#
#

# ******************************* ENVIRONMENT *******************************

NODES_2_TRAIN = [0, 1, 2, 3, 4, 5, 6, 7]
BACKGROUND_VEHICLES_ROUTE_NODES = [0, 1, 2, 3, 4, 5, 6, 7]
TIMESTEPS_LIMIT: int = 20
MODEL_PATH: str = "/home/sergi/Documents/AI_agent_env_implementation/Agents/Models/Rainbow/"
SERGI_PLOTS: str = "/home/sergi/Documents/AI_agent_env_implementation/Sergi_plots/All_nodes_8_bg_vehicles/"
SEED: int = 1976
BACKGROUND_VEHICLES = 10
# ******************************* GRAPH **************************************
# FECs coverage range
FECS_RANGE = {
    0: [(1, 0), (0, 4), (4, 5), (5, 1), (2, 1)],
    1: [(3, 2), (2, 6), (6, 7), (7, 3), (5, 6)]
}

# Manually specify scenario nodes positions for Manhattan layout
NODES_POSITION = {
    0: (0, 3), 1: (1, 3), 2: (2, 3), 3: (3, 3),
    4: (0, 2), 5: (1, 2), 6: (2, 2), 7: (3, 2),
}

# Scenario edge cost
EDGES_COST = {
    (0, 4): {'weight': 0.1},
    (5, 1): {'weight': 0.1},
    (2, 6): {'weight': 0.1},
    (7, 3): {'weight': 0.1},
    (3, 2): {'weight': 0.1},
    (2, 1): {'weight': 0.1},
    (1, 0): {'weight': 0.1},
    (4, 5): {'weight': 0.1},
    (5, 6): {'weight': 0.1},
    (6, 7): {'weight': 0.1},
}

# ***************************** VNF *******************************************

VNF_GPU: list = [2, 4]
VNF_RAM: list = [4, 8]
VNF_BW: list = [5, 25]

FEC_MAX_GPU: int = 16
FEC_MAX_RAM: int = 32
FEC_MAX_BW: int = 100

VECN_INIT: dict = {
    0: (FEC_MAX_GPU, FEC_MAX_RAM, FEC_MAX_BW),
    1: (FEC_MAX_GPU, FEC_MAX_RAM, FEC_MAX_BW),
}



