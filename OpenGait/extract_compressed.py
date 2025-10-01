import os
import argparse
import torch
import numpy as np
import yaml
from only_train_once import OTO
from modeling import models

def main():
    parser = argparse.ArgumentParser(description='Extract compressed model from checkpoint')
    parser.add_argument('--checkpoint', required=True, type=str, 
                        help='Path to the checkpoint file')
    parser.add_argument('--config', required=True, type=str, 
                        help='Path to the config file')
    parser.add_argument('--output_dir', default='./compressed_models', type=str,
                        help='Directory to save the compressed model')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as stream:
        cfgs = yaml.safe_load(stream)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Get model class from config
    model_cfg = cfgs['model_cfg']
    Model = getattr(models, model_cfg['model'])
    
    print(f"Creating {model_cfg['model']} instance")
    model = Model(cfgs, training=False)
    
    # Load checkpoint
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.cuda()
    
    # Create dummy input for the model
    print("Creating dummy input")
    batch_size = 4
    seq_len = 30  # Frames per sequence
    height = 64   # Height of silhouette 
    width = 44    # Width of silhouette
    
    # Silhouette images in batch form
    torch.manual_seed(42)  # For reproducibility
    sils = torch.rand(batch_size, seq_len, 1, height, width).cuda()
    labs = torch.zeros(batch_size).long().cuda()
    typs = torch.zeros(batch_size).long().cuda()
    vies = torch.zeros(batch_size).long().cuda()
    seqL = torch.full((batch_size,), seq_len).long().cuda()
    
    dummy_input = [sils, labs, typs, vies, seqL]
    
    # Initialize OTO
    print("Initializing OTO and constructing compressed model")
    model.oto = OTO(model=model, dummy_input=dummy_input)
    
    # Construct compressed model
    compressed_model_path = model.construct_compressed_model(out_dir=args.output_dir)
    
    if compressed_model_path:
        print(f"Compressed model saved to: {compressed_model_path}")
        
        # Load the compressed model to analyze
        compressed_model = torch.load(compressed_model_path)
        
        # Count parameters
        original_params = sum(p.numel() for p in model.parameters())
        compressed_params = sum(p.numel() for p in compressed_model.parameters())
        
        print(f"\nOriginal model parameters: {original_params:,}")
        print(f"Compressed model parameters: {compressed_params:,}")
        print(f"Compression ratio: {compressed_params/original_params:.4f}")
        print(f"Parameter reduction: {(1-compressed_params/original_params)*100:.2f}%")
        
        # Compare file sizes
        original_size_mb = os.path.getsize(args.checkpoint) / (1024*1024)
        compressed_size_mb = os.path.getsize(compressed_model_path) / (1024*1024)
        
        print(f"\nOriginal model file size: {original_size_mb:.2f} MB")
        print(f"Compressed model file size: {compressed_size_mb:.2f} MB")
        print(f"File size reduction: {(1-compressed_size_mb/original_size_mb)*100:.2f}%")
    else:
        print("Failed to construct compressed model. Check if your model has implement_compressed_model method.")

if __name__ == '__main__':
    main()