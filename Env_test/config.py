


# Environment conditions
SEED: int = 1976
TIMESTEPS_LIMIT: int = 30
BACKGROUND_VEHICLES = 0


# Cloud-aware VNF resources request values

VNF_GPU = [64, 128]
VNF_RAM = [8, 16]
VNF_BW = [1, 5]

FEC_MAX_GPU = 256
FEC_MAX_RAM = 32
FEC_MAX_BW = 100

MAX_GPU = 4
MIN_GPU = 2
MAX_RAM = 8
MIN_RAM = 4
MAX_BW = 5
MIN_BW = 1
# ********************* Environment for test **********************
FECS_RANGE = {
    1: [(1, 0), (0, 4), (4, 5), (5, 1), (2, 1), (9, 5)],
    2: [(3, 2), (2, 6), (6, 7), (7, 3), (5, 6), (11, 7)],
    3: [(8, 12), (12, 13), (13, 9), (9, 8), (4, 8), (10, 9)],
    4: [(15, 11), (11, 10), (10, 14), (14, 15), (13, 14), (6, 10)]
}

POSITION = {
    0: (0, 3), 1: (1, 3), 2: (2, 3), 3: (3, 3),
    4: (0, 2), 5: (1, 2), 6: (2, 2), 7: (3, 2),
    8: (0, 1), 9: (1, 1), 10: (2, 1), 11: (3, 1),
    12: (0, 0), 13: (1, 0), 14: (2, 0), 15: (3, 0),
}

EDGES = {
    (0, 4): {'weight': 0.1},
    (4, 8): {'weight': 0.1},
    (8, 12): {'weight': 0.1},
    (13, 9): {'weight': 0.5},
    (9, 5): {'weight': 0.5},
    (5, 1): {'weight': 0.5},
    (2, 6): {'weight': 0.5},
    (6, 10): {'weight': 0.5},
    (10, 14): {'weight': 0.5},
    (15, 11): {'weight': 0.1},
    (11, 7): {'weight': 0.1},
    (7, 3): {'weight': 0.1},
    (3, 2): {'weight': 0.1},
    (2, 1): {'weight': 0.1},
    (1, 0): {'weight': 0.1},
    (4, 5): {'weight': 0.5},
    (5, 6): {'weight': 0.5},
    (6, 7): {'weight': 0.5},
    (11, 10): {'weight': 0.5},
    (10, 9): {'weight': 0.5},
    (9, 8): {'weight': 0.5},
    (12, 13): {'weight': 0.1},
    (13, 14): {'weight': 0.1},
    (14, 15): {'weight': 0.1},
}

VECN_INIT = {
    1:  (FEC_MAX_GPU, FEC_MAX_RAM, FEC_MAX_BW),
    2:  (FEC_MAX_GPU, FEC_MAX_RAM, FEC_MAX_BW),
    3:  (FEC_MAX_GPU, FEC_MAX_RAM, FEC_MAX_BW),
    4:  (FEC_MAX_GPU, FEC_MAX_RAM, FEC_MAX_BW),
}


# NODES_2_TRAIN = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
NODES_2_TRAIN = [3, 2]