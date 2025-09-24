import torch
from only_train_once import OTO
from only_train_once.quantization.quant_model import model_to_quantize_model
from backends.simple_vit import simpleViT_cifar10
import torchvision.models
import unittest
import os

OUT_DIR = './cache'

class TestSimpleViT(unittest.TestCase):
    def test_sanity(self, dummy_input=torch.rand(1, 3, 32, 32)):
        model = simpleViT_cifar10()

        oto = OTO(model, dummy_input)
        oto.mark_unprunable_by_param_names(["to_patch_embedding.2.weight"])

        oto.visualize(view=False, out_dir=OUT_DIR, display_flops=True, display_params=True)

        # For test FLOP and param reductions. 
        full_flops = oto.compute_flops(in_million=True)['total']
        full_num_params = oto.compute_num_params(in_million=True)

        oto.random_set_zero_groups()
        oto.construct_subnet(out_dir=OUT_DIR)
        # oto.visualize(view=False, out_dir=OUT_DIR,display_flops=True, display_params=True)

        full_model = torch.load(oto.full_group_sparse_model_path)
        compressed_model = torch.load(oto.compressed_model_path)

        full_output = full_model(dummy_input)
        compressed_output = compressed_model(dummy_input)

        max_output_diff = torch.max(torch.abs(full_output - compressed_output))
        print("Maximum output difference " + str(max_output_diff.item()))
        self.assertLessEqual(max_output_diff, 1e-4)
        full_model_size = os.stat(oto.full_group_sparse_model_path)
        compressed_model_size = os.stat(oto.compressed_model_path)
        print("Size of full model     : ", full_model_size.st_size / (1024 ** 3), "GBs")
        print("Size of compress model : ", compressed_model_size.st_size / (1024 ** 3), "GBs")

        # For test FLOP and param reductions. 
        oto_compressed = OTO(compressed_model, dummy_input)
        compressed_flops = oto_compressed.compute_flops(in_million=True)['total']
        compressed_num_params = oto_compressed.compute_num_params(in_million=True)

        print("FLOP  reduction (%)    : ", 1.0 - compressed_flops / full_flops)
        print("Param reduction (%)    : ", 1.0 - compressed_num_params / full_num_params)