# Copyright (c) OpenMMLab. All rights reserved.
import copy as cp
import pickle
from typing import Dict, List, Tuple

import numpy as np
from mmcv.transforms import BaseTransform, KeyMapper
from mmengine.dataset import Compose
from mmengine.fileio import FileClient
from scipy.stats import mode
from torch.nn.modules.utils import _pair

from mmaction.registry import TRANSFORMS
from .processing import Flip, _combine_quadruple


@TRANSFORMS.register_module()
class LoadKineticsPose(BaseTransform):
    """Load Kinetics Pose given filename (The format should be pickle)

    Required keys are "filename", "total_frames", "img_shape", "frame_inds",
    "anno_inds" (for mmpose source, optional), added or modified keys are
    "keypoint", "keypoint_score".

    Args:
        io_backend (str): IO backend where frames are stored. Default: 'disk'.
        squeeze (bool): Whether to remove frames with no human pose.
            Default: True.
        max_person (int): The max number of persons in a frame. Default: 10.
        keypoint_weight (dict): The weight of keypoints. We set the confidence
            score of a person as the weighted sum of confidence scores of each
            joint. Persons with low confidence scores are dropped (if exceed
            max_person). Default: dict(face=1, torso=2, limb=3).
        source (str): The sources of the keypoints used. Choices are 'mmpose'
            and 'openpose-18'. Default: 'mmpose'.
        kwargs (dict, optional): Arguments for FileClient.
    """

    def __init__(self,
                 io_backend='disk',
                 squeeze=True,
                 max_person=100,
                 keypoint_weight=dict(face=1, torso=2, limb=3),
                 source='mmpose',
                 **kwargs):

        self.io_backend = io_backend
        self.squeeze = squeeze
        self.max_person = max_person
        self.keypoint_weight = cp.deepcopy(keypoint_weight)
        self.source = source

        if source == 'openpose-18':
            self.kpsubset = dict(
                face=[0, 14, 15, 16, 17],
                torso=[1, 2, 8, 5, 11],
                limb=[3, 4, 6, 7, 9, 10, 12, 13])
        elif source == 'mmpose':
            self.kpsubset = dict(
                face=[0, 1, 2, 3, 4],
                torso=[5, 6, 11, 12],
                limb=[7, 8, 9, 10, 13, 14, 15, 16])
        else:
            raise NotImplementedError('Unknown source of Kinetics Pose')

        self.kwargs = kwargs
        self.file_client = None

    def transform(self, results):
        """Perform the kinetics pose decoding.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        assert 'filename' in results
        filename = results.pop('filename')

        # only applicable to source == 'mmpose'
        anno_inds = None
        if 'anno_inds' in results:
            assert self.source == 'mmpose'
            anno_inds = results.pop('anno_inds')
        results.pop('box_score', None)

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        bytes = self.file_client.get(filename)

        # only the kp array is in the pickle file, each kp include x, y, score.
        kps = pickle.loads(bytes)

        total_frames = results['total_frames']

        frame_inds = results.pop('frame_inds')

        if anno_inds is not None:
            kps = kps[anno_inds]
            frame_inds = frame_inds[anno_inds]

        frame_inds = list(frame_inds)

        def mapinds(inds):
            uni = np.unique(inds)
            map_ = {x: i for i, x in enumerate(uni)}
            inds = [map_[x] for x in inds]
            return np.array(inds, dtype=np.int16)

        if self.squeeze:
            frame_inds = mapinds(frame_inds)
            total_frames = np.max(frame_inds) + 1

        # write it back
        results['total_frames'] = total_frames

        h, w = results['img_shape']
        if self.source == 'openpose-18':
            kps[:, :, 0] *= w
            kps[:, :, 1] *= h

        num_kp = kps.shape[1]
        num_person = mode(frame_inds)[-1][0]

        new_kp = np.zeros([num_person, total_frames, num_kp, 2],
                          dtype=np.float16)
        new_kpscore = np.zeros([num_person, total_frames, num_kp],
                               dtype=np.float16)
        # 32768 is enough
        num_person_frame = np.zeros([total_frames], dtype=np.int16)

        for frame_ind, kp in zip(frame_inds, kps):
            person_ind = num_person_frame[frame_ind]
            new_kp[person_ind, frame_ind] = kp[:, :2]
            new_kpscore[person_ind, frame_ind] = kp[:, 2]
            num_person_frame[frame_ind] += 1

        kpgrp = self.kpsubset
        weight = self.keypoint_weight
        results['num_person'] = num_person

        if num_person > self.max_person:
            for i in range(total_frames):
                np_frame = num_person_frame[i]
                val = new_kpscore[:np_frame, i]

                val = (
                    np.sum(val[:, kpgrp['face']], 1) * weight['face'] +
                    np.sum(val[:, kpgrp['torso']], 1) * weight['torso'] +
                    np.sum(val[:, kpgrp['limb']], 1) * weight['limb'])
                inds = sorted(range(np_frame), key=lambda x: -val[x])
                new_kpscore[:np_frame, i] = new_kpscore[inds, i]
                new_kp[:np_frame, i] = new_kp[inds, i]
            results['num_person'] = self.max_person

        results['keypoint'] = new_kp[:self.max_person]
        results['keypoint_score'] = new_kpscore[:self.max_person]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend}, '
                    f'squeeze={self.squeeze}, '
                    f'max_person={self.max_person}, '
                    f'keypoint_weight={self.keypoint_weight}, '
                    f'source={self.source}, '
                    f'kwargs={self.kwargs})')
        return repr_str


@TRANSFORMS.register_module()
class GeneratePoseTarget(BaseTransform):
    """Generate pseudo heatmaps based on joint coordinates and confidence.

    Required keys are "keypoint", "img_shape", "keypoint_score" (optional),
    added or modified keys are "imgs".

    Args:
        sigma (float): The sigma of the generated gaussian map. Default: 0.6.
        use_score (bool): Use the confidence score of keypoints as the maximum
            of the gaussian maps. Default: True.
        with_kp (bool): Generate pseudo heatmaps for keypoints. Default: True.
        with_limb (bool): Generate pseudo heatmaps for limbs. At least one of
            'with_kp' and 'with_limb' should be True. Default: False.
        skeletons (tuple[tuple]): The definition of human skeletons.
            Default: ((0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 7), (7, 9),
                      (0, 6), (6, 8), (8, 10), (5, 11), (11, 13), (13, 15),
                      (6, 12), (12, 14), (14, 16), (11, 12)),
            which is the definition of COCO-17p skeletons.
        double (bool): Output both original heatmaps and flipped heatmaps.
            Default: False.
        left_kp (tuple[int]): Indexes of left keypoints, which is used when
            flipping heatmaps. Default: (1, 3, 5, 7, 9, 11, 13, 15),
            which is left keypoints in COCO-17p.
        right_kp (tuple[int]): Indexes of right keypoints, which is used when
            flipping heatmaps. Default: (2, 4, 6, 8, 10, 12, 14, 16),
            which is right keypoints in COCO-17p.
    """

    def __init__(self,
                 sigma=0.6,
                 use_score=True,
                 with_kp=True,
                 with_limb=False,
                 skeletons=((0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 7),
                            (7, 9), (0, 6), (6, 8), (8, 10), (5, 11), (11, 13),
                            (13, 15), (6, 12), (12, 14), (14, 16), (11, 12)),
                 double=False,
                 left_kp=(1, 3, 5, 7, 9, 11, 13, 15),
                 right_kp=(2, 4, 6, 8, 10, 12, 14, 16)):

        self.sigma = sigma
        self.use_score = use_score
        self.with_kp = with_kp
        self.with_limb = with_limb
        self.double = double

        # an auxiliary const
        self.eps = 1e-4

        assert self.with_kp or self.with_limb, (
            'At least one of "with_limb" '
            'and "with_kp" should be set as True.')
        self.left_kp = left_kp
        self.right_kp = right_kp
        self.skeletons = skeletons

    def generate_a_heatmap(self, img_h, img_w, centers, sigma, max_values):
        """Generate pseudo heatmap for one keypoint in one frame.

        Args:
            img_h (int): The height of the heatmap.
            img_w (int): The width of the heatmap.
            centers (np.ndarray): The coordinates of corresponding keypoints
                (of multiple persons).
            sigma (float): The sigma of generated gaussian.
            max_values (np.ndarray): The max values of each keypoint.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        heatmap = np.zeros([img_h, img_w], dtype=np.float32)

        for center, max_value in zip(centers, max_values):
            mu_x, mu_y = center[0], center[1]
            if max_value < self.eps:
                continue

            st_x = max(int(mu_x - 3 * sigma), 0)
            ed_x = min(int(mu_x + 3 * sigma) + 1, img_w)
            st_y = max(int(mu_y - 3 * sigma), 0)
            ed_y = min(int(mu_y + 3 * sigma) + 1, img_h)
            x = np.arange(st_x, ed_x, 1, np.float32)
            y = np.arange(st_y, ed_y, 1, np.float32)

            # if the keypoint not in the heatmap coordinate system
            if not (len(x) and len(y)):
                continue
            y = y[:, None]

            patch = np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / 2 / sigma**2)
            patch = patch * max_value
            heatmap[st_y:ed_y,
                    st_x:ed_x] = np.maximum(heatmap[st_y:ed_y, st_x:ed_x],
                                            patch)

        return heatmap

    def generate_a_limb_heatmap(self, img_h, img_w, starts, ends, sigma,
                                start_values, end_values):
        """Generate pseudo heatmap for one limb in one frame.

        Args:
            img_h (int): The height of the heatmap.
            img_w (int): The width of the heatmap.
            starts (np.ndarray): The coordinates of one keypoint in the
                corresponding limbs (of multiple persons).
            ends (np.ndarray): The coordinates of the other keypoint in the
                corresponding limbs (of multiple persons).
            sigma (float): The sigma of generated gaussian.
            start_values (np.ndarray): The max values of one keypoint in the
                corresponding limbs.
            end_values (np.ndarray): The max values of the other keypoint in
                the corresponding limbs.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        heatmap = np.zeros([img_h, img_w], dtype=np.float32)

        for start, end, start_value, end_value in zip(starts, ends,
                                                      start_values,
                                                      end_values):
            value_coeff = min(start_value, end_value)
            if value_coeff < self.eps:
                continue

            min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
            min_y, max_y = min(start[1], end[1]), max(start[1], end[1])

            min_x = max(int(min_x - 3 * sigma), 0)
            max_x = min(int(max_x + 3 * sigma) + 1, img_w)
            min_y = max(int(min_y - 3 * sigma), 0)
            max_y = min(int(max_y + 3 * sigma) + 1, img_h)

            x = np.arange(min_x, max_x, 1, np.float32)
            y = np.arange(min_y, max_y, 1, np.float32)

            if not (len(x) and len(y)):
                continue

            y = y[:, None]
            x_0 = np.zeros_like(x)
            y_0 = np.zeros_like(y)

            # distance to start keypoints
            d2_start = ((x - start[0])**2 + (y - start[1])**2)

            # distance to end keypoints
            d2_end = ((x - end[0])**2 + (y - end[1])**2)

            # the distance between start and end keypoints.
            d2_ab = ((start[0] - end[0])**2 + (start[1] - end[1])**2)

            if d2_ab < 1:
                full_map = self.generate_a_heatmap(img_h, img_w, [start],
                                                   sigma, [start_value])
                heatmap = np.maximum(heatmap, full_map)
                continue

            coeff = (d2_start - d2_end + d2_ab) / 2. / d2_ab

            a_dominate = coeff <= 0
            b_dominate = coeff >= 1
            seg_dominate = 1 - a_dominate - b_dominate

            position = np.stack([x + y_0, y + x_0], axis=-1)
            projection = start + np.stack([coeff, coeff], axis=-1) * (
                end - start)
            d2_line = position - projection
            d2_line = d2_line[:, :, 0]**2 + d2_line[:, :, 1]**2
            d2_seg = (
                a_dominate * d2_start + b_dominate * d2_end +
                seg_dominate * d2_line)

            patch = np.exp(-d2_seg / 2. / sigma**2)
            patch = patch * value_coeff

            heatmap[min_y:max_y, min_x:max_x] = np.maximum(
                heatmap[min_y:max_y, min_x:max_x], patch)

        return heatmap

    def generate_heatmap(self, img_h, img_w, kps, sigma, max_values):
        """Generate pseudo heatmap for all keypoints and limbs in one frame (if
        needed).

        Args:
            img_h (int): The height of the heatmap.
            img_w (int): The width of the heatmap.
            kps (np.ndarray): The coordinates of keypoints in this frame.
            sigma (float): The sigma of generated gaussian.
            max_values (np.ndarray): The confidence score of each keypoint.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        heatmaps = []
        if self.with_kp:
            num_kp = kps.shape[1]
            for i in range(num_kp):
                heatmap = self.generate_a_heatmap(img_h, img_w, kps[:, i],
                                                  sigma, max_values[:, i])
                heatmaps.append(heatmap)

        if self.with_limb:
            for limb in self.skeletons:
                start_idx, end_idx = limb
                starts = kps[:, start_idx]
                ends = kps[:, end_idx]

                start_values = max_values[:, start_idx]
                end_values = max_values[:, end_idx]
                heatmap = self.generate_a_limb_heatmap(img_h, img_w, starts,
                                                       ends, sigma,
                                                       start_values,
                                                       end_values)
                heatmaps.append(heatmap)

        return np.stack(heatmaps, axis=-1)

    def gen_an_aug(self, results):
        """Generate pseudo heatmaps for all frames.

        Args:
            results (dict): The dictionary that contains all info of a sample.

        Returns:
            list[np.ndarray]: The generated pseudo heatmaps.
        """

        all_kps = results['keypoint']
        kp_shape = all_kps.shape

        if 'keypoint_score' in results:
            all_kpscores = results['keypoint_score']
        else:
            all_kpscores = np.ones(kp_shape[:-1], dtype=np.float32)

        img_h, img_w = results['img_shape']
        num_frame = kp_shape[1]

        imgs = []
        for i in range(num_frame):
            sigma = self.sigma
            kps = all_kps[:, i]
            kpscores = all_kpscores[:, i]

            max_values = np.ones(kpscores.shape, dtype=np.float32)
            if self.use_score:
                max_values = kpscores

            hmap = self.generate_heatmap(img_h, img_w, kps, sigma, max_values)
            imgs.append(hmap)

        return imgs

    def transform(self, results):
        """Generate pseudo heatmaps based on joint coordinates and confidence.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if not self.double:
            results['imgs'] = np.stack(self.gen_an_aug(results))
        else:
            results_ = cp.deepcopy(results)
            flip = Flip(
                flip_ratio=1, left_kp=self.left_kp, right_kp=self.right_kp)
            results_ = flip(results_)
            results['imgs'] = np.concatenate(
                [self.gen_an_aug(results),
                 self.gen_an_aug(results_)])
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'sigma={self.sigma}, '
                    f'use_score={self.use_score}, '
                    f'with_kp={self.with_kp}, '
                    f'with_limb={self.with_limb}, '
                    f'skeletons={self.skeletons}, '
                    f'double={self.double}, '
                    f'left_kp={self.left_kp}, '
                    f'right_kp={self.right_kp})')
        return repr_str


@TRANSFORMS.register_module()
class PoseCompact(BaseTransform):
    """Convert the coordinates of keypoints to make it more compact.
    Specifically, it first find a tight bounding box that surrounds all joints
    in each frame, then we expand the tight box by a given padding ratio. For
    example, if 'padding == 0.25', then the expanded box has unchanged center,
    and 1.25x width and height.

    Required keys in results are "img_shape", "keypoint", add or modified keys
    are "img_shape", "keypoint", "crop_quadruple".

    Args:
        padding (float): The padding size. Default: 0.25.
        threshold (int): The threshold for the tight bounding box. If the width
            or height of the tight bounding box is smaller than the threshold,
            we do not perform the compact operation. Default: 10.
        hw_ratio (float | tuple[float] | None): The hw_ratio of the expanded
            box. Float indicates the specific ratio and tuple indicates a
            ratio range. If set as None, it means there is no requirement on
            hw_ratio. Default: None.
        allow_imgpad (bool): Whether to allow expanding the box outside the
            image to meet the hw_ratio requirement. Default: True.

    Returns:
        type: Description of returned object.
    """

    def __init__(self,
                 padding=0.25,
                 threshold=10,
                 hw_ratio=None,
                 allow_imgpad=True):

        self.padding = padding
        self.threshold = threshold
        if hw_ratio is not None:
            hw_ratio = _pair(hw_ratio)

        self.hw_ratio = hw_ratio

        self.allow_imgpad = allow_imgpad
        assert self.padding >= 0

    def transform(self, results):
        """Convert the coordinates of keypoints to make it more compact.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        img_shape = results['img_shape']
        h, w = img_shape
        kp = results['keypoint']

        # Make NaN zero
        kp[np.isnan(kp)] = 0.
        kp_x = kp[..., 0]
        kp_y = kp[..., 1]

        min_x = np.min(kp_x[kp_x != 0], initial=np.Inf)
        min_y = np.min(kp_y[kp_y != 0], initial=np.Inf)
        max_x = np.max(kp_x[kp_x != 0], initial=-np.Inf)
        max_y = np.max(kp_y[kp_y != 0], initial=-np.Inf)

        # The compact area is too small
        if max_x - min_x < self.threshold or max_y - min_y < self.threshold:
            return results

        center = ((max_x + min_x) / 2, (max_y + min_y) / 2)
        half_width = (max_x - min_x) / 2 * (1 + self.padding)
        half_height = (max_y - min_y) / 2 * (1 + self.padding)

        if self.hw_ratio is not None:
            half_height = max(self.hw_ratio[0] * half_width, half_height)
            half_width = max(1 / self.hw_ratio[1] * half_height, half_width)

        min_x, max_x = center[0] - half_width, center[0] + half_width
        min_y, max_y = center[1] - half_height, center[1] + half_height

        # hot update
        if not self.allow_imgpad:
            min_x, min_y = int(max(0, min_x)), int(max(0, min_y))
            max_x, max_y = int(min(w, max_x)), int(min(h, max_y))
        else:
            min_x, min_y = int(min_x), int(min_y)
            max_x, max_y = int(max_x), int(max_y)

        kp_x[kp_x != 0] -= min_x
        kp_y[kp_y != 0] -= min_y

        new_shape = (max_y - min_y, max_x - min_x)
        results['img_shape'] = new_shape

        # the order is x, y, w, h (in [0, 1]), a tuple
        crop_quadruple = results.get('crop_quadruple', (0., 0., 1., 1.))
        new_crop_quadruple = (min_x / w, min_y / h, (max_x - min_x) / w,
                              (max_y - min_y) / h)
        crop_quadruple = _combine_quadruple(crop_quadruple, new_crop_quadruple)
        results['crop_quadruple'] = crop_quadruple
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(padding={self.padding}, '
                    f'threshold={self.threshold}, '
                    f'hw_ratio={self.hw_ratio}, '
                    f'allow_imgpad={self.allow_imgpad})')
        return repr_str


@TRANSFORMS.register_module()
class PreNormalize3D(BaseTransform):
    """PreNormalize for NTURGB+D 3D keypoints (x, y, z).

    PreNormalize3D first subtracts the coordinates of each joint
    from the coordinates of the 'spine' (joint #1 in ntu) of the first person
    in the first frame. Subsequently, it performs a 3D rotation to fix the Z
    axis parallel to the 3D vector from the 'hip' (joint #0) and the 'spine'
    (joint #1) and the X axis toward the 3D vector from the 'right shoulder'
    (joint #8) and the 'left shoulder' (joint #4). Codes adapted from
    https://github.com/lshiwjx/2s-AGCN.

    Required Keys:

        - keypoint
        - total_frames (optional)

    Modified Keys:

        - keypoint

    Added Keys:

        - body_center

    Args:
        zaxis (list[int]): The target Z axis for the 3D rotation.
            Defaults to ``[0, 1]``.
        xaxis (list[int]): The target X axis for the 3D rotation.
            Defaults to ``[8, 4]``.
        align_spine (bool): Whether to perform a 3D rotation to
            align the spine. Defaults to True.
        align_shoulder (bool): Whether to perform a 3D rotation
            to align the shoulder. Defaults to True.
        align_center (bool): Whether to align the body center.
            Defaults to True.
    """

    def __init__(self,
                 zaxis: List[int] = [0, 1],
                 xaxis: List[int] = [8, 4],
                 align_spine: bool = True,
                 align_shoulder: bool = True,
                 align_center: bool = True) -> None:
        self.zaxis = zaxis
        self.xaxis = xaxis
        self.align_center = align_center
        self.align_spine = align_spine
        self.align_shoulder = align_shoulder

    def unit_vector(self, vector: np.ndarray) -> np.ndarray:
        """Returns the unit vector of the vector."""
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Returns the angle in radians between vectors 'v1' and 'v2'."""
        if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
            return 0
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def rotation_matrix(self, axis: np.ndarray, theta: float) -> np.ndarray:
        """Returns the rotation matrix associated with counterclockwise
        rotation about the given axis by theta radians."""
        if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
            return np.eye(3)
        axis = np.asarray(axis)
        axis = axis / np.sqrt(np.dot(axis, axis))
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    def transform(self, results: Dict) -> Dict:
        """The transform function of :class:`PreNormalize3D`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        skeleton = results['keypoint']
        total_frames = results.get('total_frames', skeleton.shape[1])

        M, T, V, C = skeleton.shape
        assert T == total_frames
        if skeleton.sum() == 0:
            return results

        index0 = [
            i for i in range(T) if not np.all(np.isclose(skeleton[0, i], 0))
        ]

        assert M in [1, 2]
        if M == 2:
            index1 = [
                i for i in range(T)
                if not np.all(np.isclose(skeleton[1, i], 0))
            ]
            if len(index0) < len(index1):
                skeleton = skeleton[:, np.array(index1)]
                skeleton = skeleton[[1, 0]]
            else:
                skeleton = skeleton[:, np.array(index0)]
        else:
            skeleton = skeleton[:, np.array(index0)]

        T_new = skeleton.shape[1]

        if self.align_center:
            if skeleton.shape[2] == 25:
                main_body_center = skeleton[0, 0, 1].copy()
            else:
                main_body_center = skeleton[0, 0, -1].copy()
            mask = ((skeleton != 0).sum(-1) > 0)[..., None]
            skeleton = (skeleton - main_body_center) * mask

        if self.align_spine:
            joint_bottom = skeleton[0, 0, self.zaxis[0]]
            joint_top = skeleton[0, 0, self.zaxis[1]]
            axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
            angle = self.angle_between(joint_top - joint_bottom, [0, 0, 1])
            matrix_z = self.rotation_matrix(axis, angle)
            skeleton = np.einsum('abcd,kd->abck', skeleton, matrix_z)

        if self.align_shoulder:
            joint_rshoulder = skeleton[0, 0, self.xaxis[0]]
            joint_lshoulder = skeleton[0, 0, self.xaxis[1]]
            axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
            angle = self.angle_between(joint_rshoulder - joint_lshoulder,
                                       [1, 0, 0])
            matrix_x = self.rotation_matrix(axis, angle)
            skeleton = np.einsum('abcd,kd->abck', skeleton, matrix_x)

        results['keypoint'] = skeleton
        results['total_frames'] = T_new
        results['body_center'] = main_body_center
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'zaxis={self.zaxis}, '
                    f'xaxis={self.xaxis}, '
                    f'align_center={self.align_center}, '
                    f'align_spine={self.align_spine}, '
                    f'align_shoulder={self.align_shoulder})')
        return repr_str


@TRANSFORMS.register_module()
class PreNormalize2D(BaseTransform):
    """Normalize the range of keypoint values.

    Required Keys:

        - keypoint
        - img_shape (optional)

    Modified Keys:

        - keypoint

    Args:
        img_shape (tuple[int, int]): The resolution of the original video.
            Defaults to ``(1080, 1920)``.
    """

    def __init__(self, img_shape: Tuple[int, int] = (1080, 1920)) -> None:
        self.img_shape = img_shape

    def transform(self, results: Dict) -> Dict:
        """The transform function of :class:`PreNormalize2D`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        h, w = results.get('img_shape', self.img_shape)
        results['keypoint'][..., 0] = \
            (results['keypoint'][..., 0] - (w / 2)) / (w / 2)
        results['keypoint'][..., 1] = \
            (results['keypoint'][..., 1] - (h / 2)) / (h / 2)
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'img_shape={self.img_shape})')
        return repr_str


@TRANSFORMS.register_module()
class JointToBone(BaseTransform):
    """Convert the joint information to bone information.

    Required Keys:

        - keypoint

    Modified Keys:

        - keypoint

    Args:
        dataset (str): Define the type of dataset: 'nturgb+d', 'openpose',
            'coco'. Defaults to ``'nturgb+d'``.
        target (str): The target key for the bone information.
            Defaults to ``'keypoint'``.
    """

    def __init__(self,
                 dataset: str = 'nturgb+d',
                 target: str = 'keypoint') -> None:
        self.dataset = dataset
        self.target = target
        if self.dataset not in ['nturgb+d', 'openpose', 'coco']:
            raise ValueError(
                f'The dataset type {self.dataset} is not supported')
        if self.dataset == 'nturgb+d':
            self.pairs = [(0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4),
                          (6, 5), (7, 6), (8, 20), (9, 8), (10, 9), (11, 10),
                          (12, 0), (13, 12), (14, 13), (15, 14), (16, 0),
                          (17, 16), (18, 17), (19, 18), (21, 22), (20, 20),
                          (22, 7), (23, 24), (24, 11)]
        elif self.dataset == 'openpose':
            self.pairs = ((0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 1),
                          (6, 5), (7, 6), (8, 2), (9, 8), (10, 9), (11, 5),
                          (12, 11), (13, 12), (14, 0), (15, 0), (16, 14), (17,
                                                                           15))
        elif self.dataset == 'coco':
            self.pairs = ((0, 0), (1, 0), (2, 0), (3, 1), (4, 2), (5, 0),
                          (6, 0), (7, 5), (8, 6), (9, 7), (10, 8), (11, 0),
                          (12, 0), (13, 11), (14, 12), (15, 13), (16, 14))

    def transform(self, results: Dict) -> Dict:
        """The transform function of :class:`JointToBone`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        keypoint = results['keypoint']
        M, T, V, C = keypoint.shape
        bone = np.zeros((M, T, V, C), dtype=np.float32)

        assert C in [2, 3]
        for v1, v2 in self.pairs:
            bone[..., v1, :] = keypoint[..., v1, :] - keypoint[..., v2, :]
            if C == 3 and self.dataset in ['openpose', 'coco']:
                score = (keypoint[..., v1, 2] + keypoint[..., v2, 2]) / 2
                bone[..., v1, 2] = score

        results[self.target] = bone
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'dataset={self.dataset}, '
                    f'target={self.target})')
        return repr_str


@TRANSFORMS.register_module()
class ToMotion(BaseTransform):
    """Convert the joint information or bone information to corresponding
    motion information.

    Required Keys:

        - keypoint

    Added Keys:

        - motion

    Args:
        dataset (str): Define the type of dataset: 'nturgb+d', 'openpose',
            'coco'. Defaults to ``'nturgb+d'``.
        source (str): The source key for the joint or bone information.
            Defaults to ``'keypoint'``.
        target (str): The target key for the motion information.
            Defaults to ``'motion'``.
    """

    def __init__(self,
                 dataset: str = 'nturgb+d',
                 source: str = 'keypoint',
                 target: str = 'motion') -> None:
        self.dataset = dataset
        self.source = source
        self.target = target

    def transform(self, results: Dict) -> Dict:
        """The transform function of :class:`ToMotion`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        data = results[self.source]
        M, T, V, C = data.shape
        motion = np.zeros_like(data)

        assert C in [2, 3]
        motion[:, :T - 1] = np.diff(data, axis=1)
        if C == 3 and self.dataset in ['openpose', 'coco']:
            score = (data[:, :T - 1, :, 2] + data[:, 1:, :, 2]) / 2
            motion[:, :T - 1, :, 2] = score

        results[self.target] = motion

        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'dataset={self.dataset}, '
                    f'source={self.source}, '
                    f'target={self.target})')
        return repr_str


@TRANSFORMS.register_module()
class MergeSkeFeat(BaseTransform):
    """Merge multi-stream features.

    Args:
        feat_list (list[str]): The list of the keys of features.
            Defaults to ``['keypoint']``.
        target (str): The target key for the merged multi-stream information.
            Defaults to ``'keypoint'``.
        axis (int): The axis along which the features will be joined.
            Defaults to -1.
    """

    def __init__(self,
                 feat_list: List[str] = ['keypoint'],
                 target: str = 'keypoint',
                 axis: int = -1) -> None:
        self.feat_list = feat_list
        self.target = target
        self.axis = axis

    def transform(self, results: Dict) -> Dict:
        """The transform function of :class:`MergeSkeFeat`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        feats = []
        for name in self.feat_list:
            feats.append(results.pop(name))
        feats = np.concatenate(feats, axis=self.axis)
        results[self.target] = feats
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'feat_list={self.feat_list}, '
                    f'target={self.target}, '
                    f'axis={self.axis})')
        return repr_str


@TRANSFORMS.register_module()
class GenSkeFeat(BaseTransform):
    """Unified interface for generating multi-stream skeleton features.

    Required Keys:

        - keypoint
        - keypoint_score (optional)

    Args:
        dataset (str): Define the type of dataset: 'nturgb+d', 'openpose',
            'coco'. Defaults to ``'nturgb+d'``.
        feats (list[str]): The list of the keys of features.
            Defaults to ``['j']``.
        axis (int): The axis along which the features will be joined.
            Defaults to -1.
    """

    def __init__(self,
                 dataset: str = 'nturgb+d',
                 feats: List[str] = ['j'],
                 axis: int = -1) -> None:
        self.dataset = dataset
        self.feats = feats
        self.axis = axis
        ops = []
        if 'b' in feats or 'bm' in feats:
            ops.append(JointToBone(dataset=dataset, target='b'))
        ops.append(KeyMapper(remapping={'keypoint': 'j'}))
        if 'jm' in feats:
            ops.append(ToMotion(dataset=dataset, source='j', target='jm'))
        if 'bm' in feats:
            ops.append(ToMotion(dataset=dataset, source='b', target='bm'))
        ops.append(MergeSkeFeat(feat_list=feats, axis=axis))
        self.ops = Compose(ops)

    def transform(self, results: Dict) -> Dict:
        """The transform function of :class:`GenSkeFeat`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        if 'keypoint_score' in results and 'keypoint' in results:
            assert self.dataset != 'nturgb+d'
            assert results['keypoint'].shape[
                -1] == 2, 'Only 2D keypoints have keypoint_score. '
            keypoint = results.pop('keypoint')
            keypoint_score = results.pop('keypoint_score')
            results['keypoint'] = np.concatenate(
                [keypoint, keypoint_score[..., None]], -1)
        return self.ops(results)

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'dataset={self.dataset}, '
                    f'feats={self.feats}, '
                    f'axis={self.axis})')
        return repr_str


@TRANSFORMS.register_module()
class UniformSampleFrames(BaseTransform):
    """Uniformly sample frames from the video.

    To sample an n-frame clip from the video. UniformSampleFrames basically
    divide the video into n segments of equal length and randomly sample one
    frame from each segment. To make the testing results reproducible, a
    random seed is set during testing, to make the sampling results
    deterministic.

    Required Keys:

        - total_frames
        - start_index (optional)

    Added Keys:

        - frame_inds
        - frame_interval
        - num_clips
        - clip_len

    Args:
        clip_len (int): Frames of each sampled output clip.
        num_clips (int): Number of clips to be sampled. Defaults to 1.
        test_mode (bool): Store True when building test or validation dataset.
            Defaults to False.
        seed (int): The random seed used during test time. Defaults to 255.
    """

    def __init__(self,
                 clip_len: int,
                 num_clips: int = 1,
                 test_mode: bool = False,
                 seed: int = 255) -> None:
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.test_mode = test_mode
        self.seed = seed

    def _get_train_clips(self, num_frames: int, clip_len: int) -> np.ndarray:
        """Uniformly sample indices for training clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.

        Returns:
            np.ndarray: The sampled indices for training clips.
        """
        all_inds = []
        for clip_idx in range(self.num_clips):
            if num_frames < clip_len:
                start = np.random.randint(0, num_frames)
                inds = np.arange(start, start + clip_len)
            elif clip_len <= num_frames < 2 * clip_len:
                basic = np.arange(clip_len)
                inds = np.random.choice(
                    clip_len + 1, num_frames - clip_len, replace=False)
                offset = np.zeros(clip_len + 1, dtype=np.int32)
                offset[inds] = 1
                offset = np.cumsum(offset)
                inds = basic + offset[:-1]
            else:
                bids = np.array(
                    [i * num_frames // clip_len for i in range(clip_len + 1)])
                bsize = np.diff(bids)
                bst = bids[:clip_len]
                offset = np.random.randint(bsize)
                inds = bst + offset

            all_inds.append(inds)

        return np.concatenate(all_inds)

    def _get_test_clips(self, num_frames: int, clip_len: int) -> np.ndarray:
        """Uniformly sample indices for testing clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.

        Returns:
            np.ndarray: The sampled indices for testing clips.
        """

        np.random.seed(self.seed)
        all_inds = []
        for i in range(self.num_clips):
            if num_frames < clip_len:
                start_ind = i if num_frames < self.num_clips \
                    else i * num_frames // self.num_clips
                inds = np.arange(start_ind, start_ind + clip_len)
            elif clip_len <= num_frames < clip_len * 2:
                basic = np.arange(clip_len)
                inds = np.random.choice(
                    clip_len + 1, num_frames - clip_len, replace=False)
                offset = np.zeros(clip_len + 1, dtype=np.int64)
                offset[inds] = 1
                offset = np.cumsum(offset)
                inds = basic + offset[:-1]
            else:
                bids = np.array(
                    [i * num_frames // clip_len for i in range(clip_len + 1)])
                bsize = np.diff(bids)
                bst = bids[:clip_len]
                offset = np.random.randint(bsize)
                inds = bst + offset

            all_inds.append(inds)

        return np.concatenate(all_inds)

    def transform(self, results: Dict) -> Dict:
        """The transform function of :class:`UniformSampleFrames`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        num_frames = results['total_frames']

        if self.test_mode:
            inds = self._get_test_clips(num_frames, self.clip_len)
        else:
            inds = self._get_train_clips(num_frames, self.clip_len)

        inds = np.mod(inds, num_frames)
        start_index = results.get('start_index', 0)
        inds = inds + start_index

        if 'keypoint' in results:
            kp = results['keypoint']
            assert num_frames == kp.shape[1]
            num_person = kp.shape[0]
            num_persons = [num_person] * num_frames
            for i in range(num_frames):
                j = num_person - 1
                while j >= 0 and np.all(np.abs(kp[j, i]) < 1e-5):
                    j -= 1
                num_persons[i] = j + 1
            transitional = [False] * num_frames
            for i in range(1, num_frames - 1):
                if num_persons[i] != num_persons[i - 1]:
                    transitional[i] = transitional[i - 1] = True
                if num_persons[i] != num_persons[i + 1]:
                    transitional[i] = transitional[i + 1] = True
            inds_int = inds.astype(np.int)
            coeff = np.array([transitional[i] for i in inds_int])
            inds = (coeff * inds_int + (1 - coeff) * inds).astype(np.float32)

        results['frame_inds'] = inds.astype(np.int32)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = None
        results['num_clips'] = self.num_clips
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'num_clips={self.num_clips}, '
                    f'test_mode={self.test_mode}, '
                    f'seed={self.seed})')
        return repr_str


@TRANSFORMS.register_module()
class PadTo(BaseTransform):
    """Sample frames from the video.

    To sample an n-frame clip from the video, PadTo samples
    the frames from zero index, and loop or zero pad the frames
    if the length of video frames is less than the value of `length`.

    Required Keys:

        - keypoint
        - total_frames
        - start_index (optional)

    Modified Keys:

        - keypoint
        - total_frames

    Args:
        length (int): The maximum length of the sampled output clip.
        mode (str): The padding mode. Defaults to ``'loop'``.
    """

    def __init__(self, length: int, mode: str = 'loop') -> None:
        self.length = length
        assert mode in ['loop', 'zero']
        self.mode = mode

    def transform(self, results: Dict) -> Dict:
        """The transform function of :class:`PadTo`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        total_frames = results['total_frames']
        assert total_frames <= self.length
        start_index = results.get('start_index', 0)
        inds = np.arange(start_index, start_index + self.length)
        inds = np.mod(inds, total_frames)

        keypoint = results['keypoint'][:, inds].copy()
        if self.mode == 'zero':
            keypoint[:, total_frames:] = 0

        results['keypoint'] = keypoint
        results['total_frames'] = self.length
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'length={self.length}, '
                    f'mode={self.mode})')
        return repr_str


@TRANSFORMS.register_module()
class PoseDecode(BaseTransform):
    """Load and decode pose with given indices.

    Required Keys:

        - keypoint
        - total_frames (optional)
        - frame_inds (optional)
        - offset (optional)
        - keypoint_score (optional)

    Modified Keys:

        - keypoint
        - keypoint_score (optional)
    """

    def transform(self, results: Dict) -> Dict:
        """The transform function of :class:`PoseDecode`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        if 'total_frames' not in results:
            results['total_frames'] = results['keypoint'].shape[1]

        if 'frame_inds' not in results:
            results['frame_inds'] = np.arange(results['total_frames'])

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)
        frame_inds = results['frame_inds'] + offset

        results['keypoint'] = results['keypoint'][:, frame_inds].astype(
            np.float32)

        if 'keypoint_score' in results:
            kpscore = results['keypoint_score']
            results['keypoint_score'] = kpscore[:,
                                                frame_inds].astype(np.float32)

        return results

    def __repr__(self) -> str:
        repr_str = f'{self.__class__.__name__}()'
        return repr_str
