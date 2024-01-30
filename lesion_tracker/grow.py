import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, center_of_mass
import networkx as nx
import SimpleITK as sitk

# Generate a 3D segmentation mask (replace this with your actual segmentation mask)
img_1 = sitk.ReadImage('/home/melandur/Code/lesion_tracker/data/MTS39_20190822_TP1_seg.nii.gz')
img_2 = sitk.ReadImage('/home/melandur/Code/lesion_tracker/data/MTS39_20200420_TP4_seg.nii.gz')
arr_1 = sitk.GetArrayFromImage(img_1)
arr_2 = sitk.GetArrayFromImage(img_2)




segmentation_mask = arr_1

# Label connected components
labeled_mask, num_labels = label(segmentation_mask)

# Calculate the volume of each labeled region
label_volumes = np.array([np.sum(labeled_mask == label_id) for label_id in range(1, num_labels + 1)])

# Calculate the center of mass (centroid) for each labeled region
label_centroids = np.array([center_of_mass(labeled_mask == label_id) for label_id in range(1, num_labels + 1)])

# Create a graph
G = nx.Graph()

# Add nodes with volume as node weight
for label_id, volume in enumerate(label_volumes, start=1):
    G.add_node(label_id, volume=volume)

# Add edges with distance as edge weight
for i in range(num_labels):
    for j in range(i + 1, num_labels):
        distance = np.linalg.norm(label_centroids[i] - label_centroids[j])
        G.add_edge(i + 1, j + 1, distance=distance)

# Visualize the graph
pos = {i: label_centroids[i - 1][:2] for i in range(1, num_labels + 1)}  # Use only the first 2 coordinates
nx.draw(G, pos, with_labels=True, font_weight='bold')
plt.show()