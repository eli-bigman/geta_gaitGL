import torch
from only_train_once import OTO
import unittest
import os
from transformers import LlamaConfig, LlamaTokenizer
from backends import LlamaForCausalLM
from peft_lora.lora_model import LoraModel, LoraConfig

OUT_DIR = './cache'

class TestLLAMAv1LoRA(unittest.TestCase):
    def test_sanity(self, dummy_input=None):
        llama_config = LlamaConfig()
        llama_config.num_hidden_layers = 4
        llama_config.num_attention_heads = 32
        llama_config.hidden_size = 4096
        llama_config.intermediate_size = 11096
        model = LlamaForCausalLM(llama_config)

        tokenizer = LlamaTokenizer.from_pretrained('huggyllama/llama-7b')
        tokenizer.pad_token_id = (0)
        tokenizer.padding_side = "left"
        tokenizer.save_pretrained(OUT_DIR)

        text = 'This is a test sentence of a very long string and random wording that is used to test dolly model.' * 7
        input_data = tokenizer(text, return_tensors='pt').input_ids

        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj']
        lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=target_modules, lora_dropout=0.05, bias="none")
        model = LoraModel(model, lora_config)

        oto = OTO(model, dummy_input=(input_data,), strict_out_nodes=True)
        oto.visualize(out_dir=OUT_DIR)
        
        oto.random_set_zero_groups()

        oto.construct_subnet(
            merge_lora_to_base=True,
            export_huggingface_format=False,
            export_float16=False,
            full_group_sparse_model_dir=OUT_DIR,
            compressed_model_dir=OUT_DIR
        )

        text_1 = 'This is a test sentence of a very long string and random wording that is used to test dolly model.' * 7
        input_data_1 = tokenizer(text_1, return_tensors='pt').input_ids

        text_2 = 'This is a good test sentence of a pretty short string and wording that is used to test dolly model.' * 7
        input_data_2 = tokenizer(text_2, return_tensors='pt').input_ids
        
        full_model = torch.load(oto.full_group_sparse_model_path)
        compressed_model = torch.load(oto.compressed_model_path)
        full_output_1 = full_model(input_data_1.to(full_model.device))
        full_output_2 = full_model(input_data_2.to(full_model.device))
        compressed_output_1 = compressed_model(input_data_1.to(compressed_model.device))
        compressed_output_2 = compressed_model(input_data_2.to(compressed_model.device))
        max_output_diff_1 = torch.max(full_output_1.logits - compressed_output_1.logits).item()
        max_output_diff_2 = torch.max(full_output_2.logits - compressed_output_2.logits).item()
        max_output_diff_3 = torch.max(full_output_1.logits - compressed_output_2.logits).item()
        max_output_diff_4 = torch.max(full_output_2.logits - compressed_output_1.logits).item()
        print("Maximum output difference under the same inputs:")
        print(max_output_diff_1)

        print("Maximum output difference under the same inputs:")
        print(max_output_diff_2)

        print("Maximum output difference under the different inputs:")
        print(max_output_diff_3)

        print("Maximum output difference under the different inputs:")
        print(max_output_diff_4)

        full_model_size = os.stat(oto.full_group_sparse_model_path)
        compressed_model_size = os.stat(oto.compressed_model_path)
        print("Size of full model     : ", full_model_size.st_size / (1024 ** 3), "GBs")
        print("Size of compress model : ", compressed_model_size.st_size / (1024 ** 3), "GBs")

        self.assertLessEqual(max_output_diff_1, 2.0)
        self.assertLessEqual(max_output_diff_2, 2.0)
        self.assertGreaterEqual(max_output_diff_3, 2.0)
        self.assertGreaterEqual(max_output_diff_4, 2.0)
