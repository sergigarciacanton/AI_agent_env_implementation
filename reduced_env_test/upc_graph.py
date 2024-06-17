# DEPENDENCIES
from reduced_env_test.config_test import EDGES, POSITION
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple


# FUNCTIONS
def plot_graph(graph: nx.DiGraph) -> None:
    """
    Plots a directed graph with labeled nodes and weighted edges.

    Parameters:
    - graph (networkx.DiGraph): The directed graph to be plotted.
    """
    # Set the size of the plot
    plt.figure(figsize=(50, 25))

    # Draw the graph with labeled nodes and styling
    nx.draw(graph,
            pos=POSITION,
            with_labels=True,
            font_weight='bold',
            node_size=3500,
            font_size=20,
            font_color='white',
            linewidths=1.5,
            alpha=1,
            arrows=True,
            arrowsize=40)

    # Get edge labels with weights
    edge_labels: Dict[Tuple[Any, Any], Any] = nx.get_edge_attributes(graph, 'weight')

    # Manually add edge labels with styling
    for edge, weight in edge_labels.items():
        source, target = edge
        x = (POSITION[source][0] + POSITION[target][0]) / 2
        y = (POSITION[source][1] + POSITION[target][1]) / 2
        plt.text(x + 0.1, y - 0.1,
                 str(weight),
                 color='red',
                 fontsize=25,
                 ha='right',
                 va='bottom',
                 bbox=dict(boxstyle='round,pad=0.1', edgecolor='white', facecolor='white'))

    # Display the plot
    plt.show()


def get_graph() -> nx.DiGraph:
    """
    Creates and returns a directed graph with nodes, labels, and weights.

    Returns:
    - networkx.DiGraph: The created directed graph.
    """
    # Initialize a directed graph
    g = nx.DiGraph()

    # Add nodes based on the range of keys in the POSITION dictionary
    g.add_nodes_from(range(min(POSITION.keys()), max(POSITION.keys())))

    # Set labels and weights for edges
    for edge, data in EDGES.items():
        source, target = edge
        g.add_edge(source, target, weight=data['weight'])

    # Return the created graph
    return g
