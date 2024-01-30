import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist

# Create two example graphs with spatial coordinates and volumes
G1 = nx.Graph()
G1.add_node(1, pos=(1, 2, 3), volume=10)
G1.add_node(2, pos=(4, 5, 6), volume=15)
G1.add_node(3, pos=(7, 8, 9), volume=20)

G2 = nx.Graph()
G2.add_node(4, pos=(2, 3, 4), volume=12)
G2.add_node(5, pos=(5, 6, 7), volume=18)
G2.add_node(6, pos=(8, 9, 10), volume=25)

# Combine the two graphs into a single graph
G_combined = nx.disjoint_union(G1, G2)

# Calculate distances between all pairs of nodes in G_combined
all_nodes = list(G_combined.nodes)
pos_combined = {node: G_combined.nodes[node]['pos'] for node in all_nodes}
distances = cdist(np.array([pos_combined[node] for node in all_nodes]),
                  np.array([pos_combined[node] for node in all_nodes]))

# Find correspondences based on distance and volume
correspondences = {}
threshold_distance = 2.0  # Adjust this threshold based on your preference
threshold_volume = 5  # Adjust this threshold based on your preference

for node1 in G1.nodes:
    pos1 = G1.nodes[node1]['pos']
    volume1 = G1.nodes[node1]['volume']

    for node2 in G2.nodes:
        pos2 = G2.nodes[node2]['pos']
        volume2 = G2.nodes[node2]['volume']

        distance = np.linalg.norm(np.array(pos1) - np.array(pos2))

        if distance < threshold_distance and abs(volume1 - volume2) < threshold_volume:
            correspondences[node1] = node2

# Visualize the combined graph
pos_combined = {node: G_combined.nodes[node]['pos'] for node in all_nodes}
nx.draw(G_combined, pos=pos_combined, with_labels=True, font_weight='bold')

# Highlight the matched nodes
matched_nodes = set(correspondences.values())
nx.draw_networkx_nodes(G_combined, pos=pos_combined, nodelist=matched_nodes, node_color='red', node_size=300)

plt.show()

# Print the correspondences
print("Node Correspondences:", correspondences)