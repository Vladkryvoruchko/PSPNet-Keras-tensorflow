import os
import pytest

import numpy as np
from imageio import imread


def compare_2_images(validator_path, output_path):
    val_abs_path = os.path.join(os.path.dirname(__file__), validator_path)
    out_abs_path = os.path.join(os.path.dirname(__file__), output_path)
    val_img = imread(val_abs_path, pilmode='RGB')
    out_img = imread(out_abs_path, pilmode='RGB')
    assert np.all(np.equal(val_img, out_img))

def clean_test_results(output_file_no_ext):
    os.remove("tests/" + output_file_no_ext + "_probs.jpg")
    os.remove("tests/" + output_file_no_ext + "_seg.jpg")
    os.remove("tests/" + output_file_no_ext + "_seg_blended.jpg")
    os.remove("tests/" + output_file_no_ext + "_seg_read.jpg")

def test_main_flip_ade20k(cli_args_ade):
    from pspnet import main
    main(cli_args_ade)
    compare_2_images("ade20k_test_probs.jpg", "validators/ade20k_test_probs.jpg")
    compare_2_images("ade20k_test_seg.jpg", "validators/ade20k_test_seg.jpg")
    compare_2_images("ade20k_test_seg_read.jpg", "validators/ade20k_test_seg_read.jpg")
    clean_test_results("ade20k_test")


@pytest.mark.skip
def test_main_flip_cityscapes(cli_args_cityscapes):
    from pspnet import main
    main(cli_args_cityscapes)


@pytest.mark.skip
def test_main_flip_voc(cli_args_voc):
    from pspnet import main
    main(cli_args_voc)
