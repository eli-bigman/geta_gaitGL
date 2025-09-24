import torch
from only_train_once import OTO
from backends import DiffModelChurch
import unittest
import os

OUT_DIR = './cache'

class TestDiffModelChurch(unittest.TestCase):
    def test_sanity(self, dummy_input=torch.rand(1, 4, 256, 256)):
        model = DiffModelChurch()

        oto = OTO(model, dummy_input)
        # oto.visualize(view=False, out_dir=OUT_DIR, display_params=True)
        oto.random_set_zero_groups()

        oto.construct_subnet(out_dir=OUT_DIR)
    
        full_model = torch.load(oto.full_group_sparse_model_path)
        compressed_model = torch.load(oto.compressed_model_path)

        full_output = full_model(dummy_input)
        compressed_output = compressed_model(dummy_input)

        max_output_diff = torch.max(torch.abs(full_output - compressed_output))
        print("Maximum output difference " + str(max_output_diff.item()))
        # self.assertLessEqual(max_output_diff, 1e-4)
        full_model_size = os.stat(oto.full_group_sparse_model_path)
        compressed_model_size = os.stat(oto.compressed_model_path)
        print("Size of full model     : ", full_model_size.st_size / (1024 ** 3), "GBs")
        print("Size of compress model : ", compressed_model_size.st_size / (1024 ** 3), "GBs")