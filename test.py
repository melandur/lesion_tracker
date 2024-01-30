import numpy as np
from sklearn.cluster import KMeans
import SimpleITK as sitk


class KMeansWithLabels:
    def __init__(self, n_clusters=8, max_iter=300, random_state=None):
        self.kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=random_state)
        self.labels_history = []

    def fit(self, X, labels):
        # Fit k-means on the data
        self.kmeans.fit(X)

        # Get the labels assigned to each cluster
        cluster_labels = self.kmeans.labels_

        # Store the initial labels for each point
        self.labels_history.append(labels.copy())

        # Track the movement of labels during each iteration
        for _ in range(self.kmeans.n_iter_):
            new_labels = np.zeros_like(labels)
            for cluster_idx in range(self.kmeans.n_clusters):
                mask = cluster_labels == cluster_idx
                majority_label = np.argmax(np.bincount(labels[mask]))
                new_labels[mask] = majority_label

            self.labels_history.append(new_labels)

        return self

# Example usage
img_1 = sitk.ReadImage('/home/melandur/Code/lesion_tracker/data/MTS39_20190822_TP1_seg.nii.gz')
img_2 = sitk.ReadImage('/home/melandur/Code/lesion_tracker/data/MTS39_20200420_TP4_seg.nii.gz')

img_1 = sitk.BinaryThreshold(image1=img_1, lowerThreshold=1, upperThreshold=100, insideValue=1, outsideValue=0)
img_2 = sitk.BinaryThreshold(image1=img_2, lowerThreshold=1, upperThreshold=100, insideValue=1, outsideValue=0)


def get_center_point_map(image):
    connected_components = sitk.ConnectedComponent(image)
    statistics = sitk.LabelShapeStatisticsImageFilter()
    statistics.Execute(connected_components)

    centerpoint_image = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    centerpoint_image.CopyInformation(image)

    for label in range(1, len(statistics.GetLabels()) + 1):
        print(label)
        center_of_mass = statistics.GetCentroid(label)
        centerpoint_image.SetPixel(int(center_of_mass[0]), int(center_of_mass[1]), int(center_of_mass[2]), 1)

    return centerpoint_image


center_point_1 = get_center_point_map(img_1)
center_point_2 = get_center_point_map(img_2)


# initial_labels = np.array([0, 1, 0, 1, 0, 1])

arr_image1 = sitk.GetArrayFromImage(center_point_1)
arr_image2 = sitk.GetArrayFromImage(center_point_2)

points_image1 = np.argwhere(arr_image1 == 1)
points_image2 = np.argwhere(arr_image2 == 1)

kmeans_with_labels = KMeansWithLabels(n_clusters=2, random_state=42)
kmeans_with_labels.fit(points_image1, )

# Access the final cluster labels and labels history
final_cluster_labels = kmeans_with_labels.kmeans.labels_
labels_history = kmeans_with_labels.labels_history

print("Final Cluster Labels:", final_cluster_labels)
print("Labels History:")
for i, labels in enumerate(labels_history):
    print(f"Iteration {i}: {labels}")