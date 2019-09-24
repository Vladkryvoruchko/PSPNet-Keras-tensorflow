import argparse

import pytest


@pytest.fixture
def cli_args_ade():
    args = argparse.Namespace(flip=False, glob_path=None, id='0', input_path='tests/ade20k_test.jpg', input_size=500, model='pspnet50_ade20k', output_path='tests/ade20k_test.jpg', sliding=False, weights=None, multi_scale=False)
    return args

@pytest.fixture
def cli_args_cityscapes():
    args = argparse.Namespace(flip=False, glob_path=None, id='0', input_path='tests/cityscapes_test.jpg', input_size=500, model='pspnet101_cityscapes', output_path='tests/cityscapes_test.jpg', sliding=False, weights=None, multi_scale=False)
    return args

@pytest.fixture
def cli_args_voc():
    args = argparse.Namespace(flip=False, glob_path=None, id='0', input_path='tests/pascal_voc_test.jpg', input_size=500, model='pspnet101_voc2012', output_path='tests/pascal_voc_test.jpg', sliding=False, weights=None, multi_scale=False)
    return args
