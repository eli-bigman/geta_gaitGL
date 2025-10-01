import os
import torch
import argparse
from modeling import models
from only_train_once import OTO
from utils import get_msg_mgr

def extract_compressed_model():
    parser = argparse.ArgumentParser(description='Extract compressed model from GETA checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to the checkpoint file')
    parser.add_argument('--output-dir', type=str, default='./compressed_models',
                      help='Directory to save the compressed model')
    parser.add_argument('--model-name', type=str, default='GaitGLGeta',
                      help='Name of the model class (default: GaitGLGeta)')
    parser.add_argument('--cfg', type=str, default='./configs/gaitgl/gaitgl_geta.yaml',
                      help='Path to the model config file')
    parser.add_argument('--visualize', action='store_true',
                      help='Print model statistics comparison')
    args = parser.parse_args()

    msg_mgr = get_msg_mgr()
    
    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the checkpoint
    msg_mgr.log_info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Get model configuration from checkpoint
    try:
        from utils import config_loader
        cfgs = config_loader(args.cfg)
        
        # Create model instance
        msg_mgr.log_info(f"Creating {args.model_name} instance")
        ModelClass = getattr(models, args.model_name)
        model = ModelClass(cfgs, training=False)
        
        # Load weights
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
            
        model = model.cuda()
        
        # Create dummy input for the model (using same dimensions as in main_geta.py)
        msg_mgr.log_info("Creating dummy input for model tracing")
        batch_size = 4
        seq_len = 30  # Frames per sequence
        height = 64   # Height of silhouette
        width = 44    # Width of silhouette
        
        torch.manual_seed(42)  # For reproducibility
        sils = torch.rand(batch_size, seq_len, 1, height, width).cuda()
        labs = torch.zeros(batch_size).long().cuda()
        typs = torch.zeros(batch_size).long().cuda()
        vies = torch.zeros(batch_size).long().cuda()
        seqL = torch.full((batch_size,), seq_len).long().cuda()
        
        dummy_input = [sils, labs, typs, vies, seqL]

        # Create OTO instance and construct compressed model
        msg_mgr.log_info("Initializing OTO and constructing compressed model")
        model.eval()  # Set to evaluation mode
        model.oto = OTO(model=model, dummy_input=dummy_input)
        compressed_model_path = model.construct_compressed_model(out_dir=args.output_dir)
        
        if compressed_model_path:
            msg_mgr.log_info(f"✓ Compressed model saved to: {compressed_model_path}")
        else:
            msg_mgr.log_warning("⚠ No compressed model was created, please check if GETA/HESSO was used during training")
            return

        # Compare models if visualize flag is set
        if args.visualize and compressed_model_path:
            # Load the compressed model
            compressed_model = torch.load(compressed_model_path)
            
            # Count parameters
            original_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            compressed_params = sum(p.numel() for p in compressed_model.parameters() if p.requires_grad)
            
            msg_mgr.log_info("\n======= MODEL STATISTICS =======")
            msg_mgr.log_info(f"Original model parameters: {original_params:,}")
            msg_mgr.log_info(f"Compressed model parameters: {compressed_params:,}")
            msg_mgr.log_info(f"Compression ratio: {compressed_params/original_params:.4f}")
            msg_mgr.log_info(f"Parameter reduction: {(1-compressed_params/original_params)*100:.2f}%")
            
            # Calculate file sizes
            checkpoint_size_mb = os.path.getsize(args.checkpoint) / (1024*1024)
            compressed_size_mb = os.path.getsize(compressed_model_path) / (1024*1024)
            
            msg_mgr.log_info(f"\nOriginal checkpoint size: {checkpoint_size_mb:.2f} MB")
            msg_mgr.log_info(f"Compressed model size: {compressed_size_mb:.2f} MB")
            msg_mgr.log_info(f"File size reduction: {(1-compressed_size_mb/checkpoint_size_mb)*100:.2f}%")
            
            # Try to compute MACs/FLOPs if supported by OTO
            try:
                oto_original = OTO(model=model, dummy_input=dummy_input)
                oto_compressed = OTO(model=compressed_model, dummy_input=dummy_input)
                
                # Calculate MACs (Multiply-Accumulate Operations)
                original_macs = oto_original.compute_macs(in_million=True)['total']
                compressed_macs = oto_compressed.compute_macs(in_million=True)['total']
                
                msg_mgr.log_info(f"\nOriginal model MACs: {original_macs:.2f}M")
                msg_mgr.log_info(f"Compressed model MACs: {compressed_macs:.2f}M")
                msg_mgr.log_info(f"MACs reduction: {(1-compressed_macs/original_macs)*100:.2f}%")
                msg_mgr.log_info("==================================")
            except Exception as e:
                msg_mgr.log_warning(f"Could not compute MACs: {e}")
            
    except Exception as e:
        msg_mgr.log_error(f"Error extracting compressed model: {str(e)}")
        raise

if __name__ == "__main__":
    extract_compressed_model()