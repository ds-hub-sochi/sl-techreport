# Copyright (c) OpenMMLab. All rights reserved.
import copy
import unittest

import numpy as np
import pytest
import torch
from mmengine.structures import InstanceData, LabelData
from mmengine.testing import assert_dict_has_keys
from numpy.testing import assert_array_equal

from mmaction.datasets.transforms import (FormatAudioShape, FormatGCNInput,
                                          FormatShape, PackActionInputs,
                                          Transpose)
from mmaction.registry import TRANSFORMS
from mmaction.structures import ActionDataSample
from mmaction.utils import register_all_modules

register_all_modules()


class TestPackActionInputs(unittest.TestCase):

    def test_transform(self):
        # keypoint input
        results = dict(keypoint=np.random.randn(1, 2, 300, 17, 3), label=1)
        transform = PackActionInputs()
        results = transform(results)
        self.assertIn('inputs', results)
        self.assertIn('data_samples', results)
        self.assertIsInstance(results['inputs'], torch.Tensor)
        self.assertEqual(results['data_samples'].gt_labels.item,
                         torch.LongTensor([1]))

        # audio input
        data = dict(
            audios=np.random.randn(3, 1, 128, 80),
            label=[1],
            filename='test.txt')

        cfg = dict(type='PackActionInputs')
        transform = TRANSFORMS.build(cfg)
        results = transform(copy.deepcopy(data))
        self.assertIn('inputs', results)
        self.assertIsInstance(results['inputs'], torch.Tensor)
        # img input with label
        data = dict(
            imgs=np.random.randn(256, 256, 3),
            label=[1],
            filename='test.txt',
            original_shape=(256, 256, 3),
            img_shape=(256, 256, 3),
            pad_shape=(256, 256, 3),
            flip_direction='vertical')

        cfg = dict(type='PackActionInputs')
        transform = TRANSFORMS.build(cfg)
        results = transform(copy.deepcopy(data))
        self.assertIn('inputs', results)
        self.assertIsInstance(results['inputs'], torch.Tensor)
        self.assertIn('data_samples', results)
        self.assertIsInstance(results['data_samples'], ActionDataSample)
        self.assertIn('img_shape', results['data_samples'].metainfo_keys())
        self.assertIsInstance(results['data_samples'].gt_labels, LabelData)

        # Test grayscale image
        data['imgs'] = data['imgs'].mean(-1)
        results = transform(copy.deepcopy(data))
        self.assertIn('inputs', results)
        self.assertIsInstance(results['inputs'], torch.Tensor)
        self.assertEqual(results['inputs'].shape, (256, 256))

        # Test without `img` and `gt_label`
        del data['imgs']
        del data['label']
        with self.assertRaises(ValueError):
            results = transform(copy.deepcopy(data))
            self.assertNotIn('gt_labels', results['data_samples'])

        # img input with gt_bboxes
        data = dict(
            imgs=np.random.randn(256, 256, 3),
            gt_bboxes=np.array([[0, 0, 340, 224]]),
            gt_labels=[1],
            proposals=np.array([[0, 0, 340, 224]]),
            filename='test.txt')

        cfg = dict(type='PackActionInputs')
        transform = TRANSFORMS.build(cfg)
        results = transform(copy.deepcopy(data))
        self.assertIn('inputs', results)
        self.assertIsInstance(results['inputs'], torch.Tensor)
        self.assertIn('data_samples', results)
        self.assertIsInstance(results['data_samples'], ActionDataSample)
        self.assertIsInstance(results['data_samples'].gt_instances,
                              InstanceData)
        self.assertIsInstance(results['data_samples'].proposals, InstanceData)

    def test_repr(self):
        cfg = dict(
            type='PackActionInputs', meta_keys=['flip_direction', 'img_shape'])
        transform = TRANSFORMS.build(cfg)
        self.assertEqual(
            repr(transform),
            "PackActionInputs(meta_keys=['flip_direction', 'img_shape'])")


class TestPackLocalizationInputs(unittest.TestCase):

    def test_transform(self):
        # raw_feature input
        data = dict(
            raw_feature=np.random.randn(400, 5),
            gt_bbox=np.array([[0.1, 0.3], [0.375, 0.625]]),
            filename='test.txt')

        cfg = dict(type='PackLocalizationInputs', keys='gt_bbox')
        transform = TRANSFORMS.build(cfg)
        results = transform(copy.deepcopy(data))
        self.assertIn('inputs', results)
        self.assertIsInstance(results['inputs'], torch.Tensor)
        self.assertIn('data_samples', results)
        self.assertIsInstance(results['data_samples'], ActionDataSample)
        self.assertIsInstance(results['data_samples'].gt_instances,
                              InstanceData)

        del data['raw_feature']
        with self.assertRaises(ValueError):
            transform(copy.deepcopy(data))

        # bsp_feature input
        data['bsp_feature'] = np.random.randn(100, 32)
        results = transform(copy.deepcopy(data))
        self.assertIn('inputs', results)
        self.assertIsInstance(results['inputs'], torch.Tensor)
        self.assertIn('data_samples', results)
        self.assertIsInstance(results['data_samples'], ActionDataSample)
        self.assertIsInstance(results['data_samples'].gt_instances,
                              InstanceData)

    def test_repr(self):
        cfg = dict(
            type='PackLocalizationInputs',
            meta_keys=['video_name', 'feature_frame'])
        transform = TRANSFORMS.build(cfg)
        self.assertEqual(
            repr(transform),
            "PackLocalizationInputs(meta_keys=['video_name', 'feature_frame'])"
        )


def test_transpose():
    results = dict(imgs=np.random.randn(256, 256, 3))
    keys = ['imgs']
    order = [2, 0, 1]
    transpose = Transpose(keys, order)
    results = transpose(results)
    assert results['imgs'].shape == (3, 256, 256)
    assert repr(transpose) == transpose.__class__.__name__ + \
        f'(keys={keys}, order={order})'


def test_format_shape():
    with pytest.raises(ValueError):
        # invalid input format
        FormatShape('NHWC')

    # 'NCHW' input format
    results = dict(
        imgs=np.random.randn(3, 224, 224, 3), num_clips=1, clip_len=3)
    format_shape = FormatShape('NCHW')
    assert format_shape(results)['input_shape'] == (3, 3, 224, 224)

    # `NCTHW` input format with num_clips=1, clip_len=3
    results = dict(
        imgs=np.random.randn(3, 224, 224, 3), num_clips=1, clip_len=3)
    format_shape = FormatShape('NCTHW')
    assert format_shape(results)['input_shape'] == (1, 3, 3, 224, 224)

    # `NCTHW` input format with num_clips=2, clip_len=3
    results = dict(
        imgs=np.random.randn(18, 224, 224, 3), num_clips=2, clip_len=3)
    assert format_shape(results)['input_shape'] == (6, 3, 3, 224, 224)
    target_keys = ['imgs', 'input_shape']
    assert assert_dict_has_keys(results, target_keys)

    assert repr(format_shape) == format_shape.__class__.__name__ + \
        "(input_format='NCTHW')"

    # `NCHW_Flow` input format
    results = dict(imgs=np.random.randn(6, 224, 224), num_clips=1, clip_len=3)
    format_shape = FormatShape('NCHW_Flow')
    assert format_shape(results)['input_shape'] == (1, 6, 224, 224)

    # `NPTCHW` input format
    results = dict(
        imgs=np.random.randn(72, 224, 224, 3),
        num_clips=9,
        clip_len=1,
        num_proposals=8)
    format_shape = FormatShape('NPTCHW')
    assert format_shape(results)['input_shape'] == (8, 9, 3, 224, 224)


def test_format_audio_shape():
    with pytest.raises(ValueError):
        # invalid input format
        FormatAudioShape('XXXX')

    # `NCTF` input format
    results = dict(audios=np.random.randn(3, 128, 8))
    format_shape = FormatAudioShape('NCTF')
    assert format_shape(results)['input_shape'] == (3, 1, 128, 8)
    assert repr(format_shape) == format_shape.__class__.__name__ + \
        "(input_format='NCTF')"


def test_format_gcn_input():
    with pytest.raises(AssertionError):
        FormatGCNInput(mode='invalid')

    results = dict(
        keypoint=np.random.randn(2, 10, 17, 2),
        keypoint_score=np.random.randn(2, 10, 17))
    format_shape = FormatGCNInput(num_person=2, mode='zero')
    results = format_shape(results)
    assert results['keypoint'].shape == (1, 2, 10, 17, 3)
    assert repr(format_shape) == 'FormatGCNInput(num_person=2, mode=zero)'

    results = dict(keypoint=np.random.randn(2, 40, 25, 3), num_clips=4)
    format_shape = FormatGCNInput(num_person=2, mode='zero')
    results = format_shape(results)
    assert results['keypoint'].shape == (4, 2, 10, 25, 3)

    results = dict(keypoint=np.random.randn(1, 10, 25, 3))
    format_shape = FormatGCNInput(num_person=2, mode='zero')
    results = format_shape(results)
    assert results['keypoint'].shape == (1, 2, 10, 25, 3)
    assert_array_equal(results['keypoint'][:, 1], np.zeros((1, 10, 25, 3)))

    results = dict(keypoint=np.random.randn(1, 10, 25, 3))
    format_shape = FormatGCNInput(num_person=2, mode='loop')
    results = format_shape(results)
    assert results['keypoint'].shape == (1, 2, 10, 25, 3)
    assert_array_equal(results['keypoint'][:, 1], results['keypoint'][:, 0])

    results = dict(keypoint=np.random.randn(3, 10, 25, 3))
    format_shape = FormatGCNInput(num_person=2, mode='zero')
    results = format_shape(results)
    assert results['keypoint'].shape == (1, 2, 10, 25, 3)
