# DEPENDENCIES
from config import EDGES_COST, NODES_POSITION
import networkx as nx
import matplotlib.pyplot as plt

NODE_SIZE = 5500
FONT_SIZE = 30


def plot_graph(graph,
               node_size=NODE_SIZE,
               font_size=FONT_SIZE,
               font_color='white',
               linewidths=1.5,
               alpha=1,
               arrows=True,
               arrowsize=40,
               edge_width=3,
               ):
    # Figure size
    plt.figure(figsize=(20, 20))

    # Draw the graph with the scaled positions
    nx.draw(graph,
            pos=NODES_POSITION,
            with_labels=True,
            font_weight='bold',
            node_size=node_size,
            font_size=font_size,
            font_color=font_color,
            linewidths=linewidths,
            alpha=alpha,
            arrows=arrows,
            arrowsize=arrowsize,
            width=edge_width,
            )

    # Edge labels
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, NODES_POSITION, edge_labels=edge_labels, font_color='red', font_size=25)

    plt.show()
    #  plt.savefig('/home/carlos/P.h.D_Carlos/Barcelona/Figures/bcn_eixample.jpg', format='jpeg')


def get_graph():
    g = nx.DiGraph()
    # Add nodes
    g.add_nodes_from(range(min(NODES_POSITION.keys()), max(NODES_POSITION.keys())))
    # Set labels and weights
    for edge, data in EDGES_COST.items():
        source, target = edge
        g.add_edge(source, target, weight=data['weight'])

    return g
