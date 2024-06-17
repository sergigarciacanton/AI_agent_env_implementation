# Environment conditions
NODES_2_TRAIN: list = [0, 1, 2, 3, 4, 5, 6, 7]
BACKGROUND_VEHICLES_ROUTE_NODES = [0, 1, 2, 3, 4, 5, 6, 7]
SEED: int = 1976
TIMESTEPS_LIMIT: int = 20
BACKGROUND_VEHICLES: int = 0
MODEL_PATH: str = "/home/user/Documents/AI_agent_env_implementation/Agents/Models/Rainbow/"
SERGI_PLOTS: str = "/home/user/Documents/AI_agent_env_implementation/Sergi_plots/All_nodes_8_bg_vehicles/"

# Cloud-aware VNF resources request values

VNF_GPU: list = [2, 4]
VNF_RAM: list = [4, 8]
VNF_BW: list = [5, 25]

FEC_MAX_GPU: int = 16
FEC_MAX_RAM: int = 32
FEC_MAX_BW: int = 100

# ********************* Environment for test **********************
FECS_RANGE: dict = {
    0: [(1, 0), (0, 4), (4, 5), (5, 1), (2, 1)],
    1: [(3, 2), (2, 6), (6, 7), (7, 3), (5, 6)]
}

POSITION: dict = {
    0: (0, 3), 1: (1, 3), 2: (2, 3), 3: (3, 3),
    4: (0, 2), 5: (1, 2), 6: (2, 2), 7: (3, 2)
}

EDGES: dict = {
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

VECN_INIT: dict = {
    0: (FEC_MAX_GPU, FEC_MAX_RAM, FEC_MAX_BW),
    1: (FEC_MAX_GPU, FEC_MAX_RAM, FEC_MAX_BW)
}
