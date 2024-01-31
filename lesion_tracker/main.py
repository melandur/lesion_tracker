import copy

import SimpleITK as sitk
from scipy.ndimage import rotate
import numpy as np
from loguru import logger


class Subject:

    def __init__(self, src):
        self.store = {}
        self.seg_mask_sitk = sitk.ReadImage(src)
        self.connected_componentes()
        self.centroids()

    def __iter__(self) -> tuple[int, dict]:
        for id, values in self.store.items():
            yield id, values

    def __len__(self):
        return len(self.store)

    def connected_componentes(self):
        """Get the connected components of the image"""

        cc_filter = sitk.ConnectedComponentImageFilter()
        cc_filter.SetFullyConnected(True)  # Setting to True is less restrictive, gives fewer connected components

        binary_img_sitk = sitk.BinaryThreshold(image1=self.seg_mask_sitk, lowerThreshold=1)
        lesions = cc_filter.Execute(binary_img_sitk)  # connected components

        rl_filter = sitk.RelabelComponentImageFilter()
        lesions = rl_filter.Execute(lesions)  # sort lesions by size

        if rl_filter.GetNumberOfObjects() == 0:
            logger.warning('No connected components found')

        for label, _ in enumerate(rl_filter.GetSizeOfObjectsInPixels(), start=1):
            component = sitk.BinaryThreshold(lesions, label, label, 1, 0)
            self.store[label] = {'component_sitk': component}

    def centroids(self):
        """Calculate the center of mass of the image"""

        for id, values in self.store.items():
            component = values['component_sitk']
            statistics = sitk.LabelShapeStatisticsImageFilter()
            statistics.Execute(component)

            spherical_radius = int(statistics.GetEquivalentSphericalRadius(label=1))

            centroid_img_sitk = sitk.Image(component.GetSize(), sitk.sitkUInt8)
            centroid_img_sitk.CopyInformation(component)

            center_point = statistics.GetCentroid(label=1)
            center_point = tuple([int(point) for point in center_point])
            centroid_value = 1
            centroid_img_sitk.SetPixel(center_point[0],
                                       center_point[1],
                                       center_point[2],
                                       centroid_value)

            self.store[id].update({
                'centroid_sitk': centroid_img_sitk,
                'center_point': center_point,
                'radius': spherical_radius,
                'context': {
                    'matches': [],
                    'verdict': None,
                    'label': None}
            })


class ExpansiveMatching:

    def __init__(self, subject_1: type(Subject), subject_2: type(Subject)):
        self.subject_1 = subject_1
        self.subject_2 = subject_2

    def __call__(self) -> tuple[type(Subject), type(Subject)]:

        for id_1, data_1 in self.subject_1:

            centroid_1_sitk = data_1['centroid_sitk']
            radius_1 = data_1['radius']

            for id_2, data_2 in self.subject_2:

                centroid_2_sitk = data_2['centroid_sitk']
                radius_2 = data_2['radius']

                for radius in range(0, 100):
                    expansion_radius_1, reached_limit_1 = self.expand(centroid_1_sitk, radius_1, radius, 1)
                    expansion_radius_2, reached_limit_2 = self.expand(centroid_2_sitk, radius_2, radius, 1)

                    if reached_limit_1 and reached_limit_2:
                        break

                    dil_img_1_sitk = self.dilate_image(centroid_1_sitk, radius)
                    dil_img_2_sitk = self.dilate_image(centroid_2_sitk, radius)

                    if dil_img_1_sitk and dil_img_2_sitk:
                        overlap = sitk.And(dil_img_1_sitk, dil_img_2_sitk)
                        if sitk.GetArrayFromImage(overlap).any():
                            if id_2 not in self.subject_1.store[id_1]['context']['matches']:
                                self.subject_1.store[id_1]['context']['matches'].append(id_2)
                                self.subject_1.store[id_1]['context']['match_radius'] = expansion_radius_1
                            if id_1 not in self.subject_2.store[id_2]['context']['matches']:
                                self.subject_2.store[id_2]['context']['matches'].append(id_1)
                                self.subject_2.store[id_2]['context']['match_radius'] = expansion_radius_2

        return self.subject_1, self.subject_2

    @staticmethod
    def expand(img_sitk: sitk.Image, native_radius: int, radius: int, exp_factor: float = 1.0) -> tuple[float, bool]:
        """Calculate the radius of the lesion and limit the radius if it is too large"""

        statistics = sitk.LabelShapeStatisticsImageFilter()
        statistics.Execute(img_sitk)
        if radius > native_radius * exp_factor:
            return native_radius * exp_factor, True
        return radius, False

    @staticmethod
    def dilate_image(centroid_sitk: sitk.Image, radius: int) -> sitk.Image | None:
        """Dilate the image with the centroid as the center of the dilation"""

        dilate = sitk.BinaryDilateImageFilter()
        dilate.SetKernelRadius(radius)
        dilate.SetForegroundValue(1)
        dilated_img_sitk = dilate.Execute(centroid_sitk)
        return dilated_img_sitk


class Analyser:

    def __init__(self, subject_1: type(Subject), subject_2: type(Subject)):
        self.subject_1 = subject_1
        self.subject_2 = subject_2

    def __call__(self):
        """Analyse the matches and find the best match"""

        for id_1, data_1 in self.subject_1:
            if len(data_1['context']['matches']) == 1:
                self.subject_1.store[id_1]['context']['verdict'] = 'match'
                self.subject_1.store[id_1]['context']['label'] = id_1

        for id_2, data_2 in self.subject_2:
            if len(data_2['context']['matches']) == 1:
                self.subject_2.store[id_2]['context']['verdict'] = 'match'
                self.subject_2.store[id_2]['context']['label'] = data_2['context']['matches'][0]

        for id_1, data_1 in self.subject_1:
            if len(data_1['context']['matches']) == 0:
                self.subject_1.store[id_1]['context']['verdict'] = 'missed'
                self.subject_1.store[id_1]['context']['label'] = id_1

        new_label = len(self.subject_1)
        for id_2, data_2 in self.subject_2:
            if len(data_2['context']['matches']) == 0:
                new_label += 1
                self.subject_2.store[id_2]['context']['verdict'] = 'new'
                self.subject_2.store[id_2]['context']['label'] = new_label

        return self.subject_1, self.subject_2


class Exporter:

    def __init__(self, subject_1: type(Subject), subject_2: type(Subject)):
        self.subject_1 = subject_1
        self.subject_2 = subject_2

    def merge_components(self, subject: type(Subject)) -> sitk.Image:
        """Merge the components of the subject into a single image with correct labels"""

        img_sitk = None
        for id, data in subject:
            component_sitk = data['component_sitk']
            component_sitk = self.change_label_of_lesion(component_sitk, data['context']['label'])
            if img_sitk is None:
                img_sitk = component_sitk
            else:
                img_sitk = sitk.Add(img_sitk, component_sitk)
        return img_sitk

    def __call__(self, relabel_subject_1: bool = False):

        if relabel_subject_1:
            img_1_sitk = self.merge_components(self.subject_1)
            sitk.WriteImage(img_1_sitk, '/home/melandur/tmp/img_1.nii.gz')

        img_2_sitk = self.merge_components(self.subject_2)
        sitk.WriteImage(img_2_sitk, '/home/melandur/tmp/img_2.nii.gz')

    def get_zero_mask(self, image):
        """Get a zero mask of the image"""

        zero_mask = sitk.Image(image.GetSize(), sitk.sitkUInt8)
        zero_mask.CopyInformation(image)
        return zero_mask

    @staticmethod
    def change_label_of_lesion(image, new_label):
        """Change the label of the lesion"""

        img_arr = sitk.GetArrayFromImage(image)
        tmp_arr = copy.deepcopy(img_arr)
        img_arr[tmp_arr == 1] = new_label
        new_img_sitk = sitk.GetImageFromArray(img_arr)
        new_img_sitk.CopyInformation(image)
        return new_img_sitk

    @staticmethod
    def manipulate_img(img: sitk.Image, angles: tuple = (0, 0, 0), shifts: tuple = (0, 0, 0)) -> sitk.Image:
        """Modify the image by rotation and shifting"""

        matrix = sitk.GetArrayFromImage(img)

        for angle, axis in zip(angles, ((0, 1), (0, 2), (1, 2))):
            matrix = rotate(matrix, angle=angle, axes=axis, reshape=False)

        for shift, axis in zip(shifts, (0, 1, 2)):
            matrix = np.roll(matrix, shift=shift, axis=axis)

        rot_img = sitk.GetImageFromArray(matrix)
        rot_img.CopyInformation(img)
        return rot_img


if __name__ == '__main__':
    s_1 = Subject('/home/melandur/Code/lesion_tracker/data/MTS39_20190822_TP1_seg.nii.gz')
    s_2 = Subject('/home/melandur/Code/lesion_tracker/data/MTS39_20200420_TP4_seg.nii.gz')

    exp_match = ExpansiveMatching(s_1, s_2)
    s_1, s_2 = exp_match()

    analyser = Analyser(s_1, s_2)
    s_1, s_2 = analyser()

    print(s_1.store)
    print(s_2.store)

    exporter = Exporter(s_1, s_2)
    exporter(relabel_subject_1=True)
