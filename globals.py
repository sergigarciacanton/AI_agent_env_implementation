

# ******************************* ENVIRONMENT *******************************
# CAVs in the system
NUM_CAVS: int = 1

# ******************************* GRAPH **************************************
# FECs coverage range
FECS_RANGE = {
    0: [1, 2, 5, 6, 9, 10],
    1: [2, 3, 4, 6, 7, 8],
    2: [9, 10, 11, 13, 14, 15],
    3: [7, 8, 11, 12, 15, 16]
}

# Manually specify scenario nodes positions for Manhattan layout
NODES_POSITION = {
    1: (0, 3), 2: (1, 3), 3: (2, 3), 4: (3, 3),
    5: (0, 2), 6: (1, 2), 7: (2, 2), 8: (3, 2),
    9: (0, 1), 10: (1, 1), 11: (2, 1), 12: (3, 1),
    13: (0, 0), 14: (1, 0), 15: (2, 0), 16: (3, 0),
}

# Scenario edge cost
EDGES_COST = {
    # row 1
    (1, 2): 5,
    (2, 3): 5,
    (3, 4): 5,
    # row 2
    (5, 6): 5,
    (6, 7): 1,
    (7, 8): 5,
    # row 3
    (9, 10): 1,
    (10, 11): 5,
    (11, 12): 1,
    # row 4
    (13, 14): 5,
    (14, 15): 1,
    (15, 16): 5,
    # column 1
    (1, 5): 5,
    (5, 9): 5,
    (9, 13): 5,
    # column 2
    (2, 6): 1,
    (6, 10): 1,
    (10, 14): 1,
    # column 3
    (3, 7): 1,
    (7, 11): 9,
    (11, 15): 1,
    # column 4
    (4, 8): 1,
    (8, 12): 5,
    (12, 16): 1,
}


# ***************************** SOCKETS ***************************************
# Port where each FEC listens for CAV connections
FEC_PORT = 5010
# IP address where FECs connect to control
CONTROL_IP_PUBLIC = "147.83.113.192"
# Port where FECs connect to control
CONTROL_PORT_PUBLIC = 30130
# Actual IP address of control server
CONTROL_IP_PRIVATE = "10.233.101.156"
# Actual port where control listens
CONTROL_PORT_PRIVATE = 5000


# ***************************** VNF *******************************************
MIN_GPU = 2
MAX_GPU = 4
MIN_RAM = 4
MAX_RAM = 8
MIN_BW = 1
MAX_BW = 5

# *******************************AP********************************************
# Interface name of Wi-Fi card for FECs
WLAN_IF_NAME = "wlan0"
# SSID name of Wi-Fi network for ALL FECs (must be the same for all!)
WLAN_SSID_NAME = "Test301"
# Wi-Fi network's password for ALL FECs (must be the same for all!)
WLAN_PASSWORD = "1234567890"
# IP address of each FEC on its Wi-Fi interface
WLAN_AP_IP = "10.0.0.1"
# Netmask of each FECs Wi-Fi interface
WLAN_NETMASK = "255.255.255.0"
# Interface name of ethernet card for FECs
ETH_IF_NAME = "eth0"

# ******************************RABBIT*****************************************
# Username for rabbitMQ server
RABBIT_USERNAME = "sergi"
# Password for rabbitMQ server
RABBIT_PASSWORD = "EETAC2023"
# Exchange name where to publish/subscribe
RABBIT_EXCHANGE_NAME = "test"
# Port where FECs have to subscribe to get broadcast information
RABBIT_PORT = 30128
