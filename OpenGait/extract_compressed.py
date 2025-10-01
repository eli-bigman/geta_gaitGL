import os
import argparse
import torch
import numpy as np
import yaml
import sys

# Add necessary paths
current_dir = os.path.dirname(os.path.abspath(__file__))
opengait_dir = os.path.join(current_dir, 'opengait')
sys.path.append(current_dir)
sys.path.append(opengait_dir)

# Initialize distributed environment to prevent errors
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    
# Initialize distributed process group if not already initialized
# This fixes "Default process group has not been initialized" error
try:
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method='env://',
            world_size=1,
            rank=0
        )
except ValueError:
    # If environment variables are not set, use file-based initialization
    try:
        torch.distributed.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method='file:///tmp/pytorch_distributed_init',
            world_size=1,
            rank=0
        )
    except Exception as e:
        # Get msg_mgr early - this is fine even before full initialization
        msg_mgr = get_msg_mgr()
        msg_mgr.log_warning(f"Warning: Could not initialize distributed environment: {e}")
        msg_mgr.log_info("Setting up dummy distributed environment...")
        # Monkey patch distributed functions to avoid errors
        original_get_rank = torch.distributed.get_rank
        torch.distributed.get_rank = lambda: 0
        original_get_world_size = torch.distributed.get_world_size
        torch.distributed.get_world_size = lambda: 1

# Now import from the correct paths
from opengait.only_train_once import OTO
from opengait.modeling import models
from opengait.utils import get_msg_mgr

def main():
    parser = argparse.ArgumentParser(description='Extract compressed model from checkpoint')
    parser.add_argument('--checkpoint', required=True, type=str, 
                        help='Path to the checkpoint file')
    parser.add_argument('--config', required=True, type=str, 
                        help='Path to the config file')
    parser.add_argument('--output_dir', default='./compressed_models', type=str,
                        help='Directory to save the compressed model')
    parser.add_argument('--log_to_file', action='store_true',
                        help='Log to file, default path is the output directory')
    args = parser.parse_args()
    
    # Initialize message manager
    msg_mgr = get_msg_mgr()
    output_path = os.path.dirname(args.output_dir)
    msg_mgr.init_logger(output_path, args.log_to_file)

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
    
    # Let's try to patch the BaseModel.__init__ method to bypass the transform
    # This is a more direct approach than trying to guess the right transform type
    import opengait.modeling.base_model as base_model
    original_init = base_model.BaseModel.__init__
    
    def patched_init(self, *args, **kwargs):
        try:
            original_init(self, *args, **kwargs)
        except KeyError as e:
            if str(e) == "'transform'" or "transform" in str(e):
                print("Bypassing transform error in model initialization")
                # Set necessary attributes to avoid errors
                self.msg_mgr = base_model.get_msg_mgr()
                self.cfgs = args[0]
                self.engine_cfg = self.cfgs['trainer_cfg'] if kwargs.get('training', False) else self.cfgs.get('evaluator_cfg', {})
                # Skip the transform part
                self.trainer_trfs = None
            else:
                raise
    
    # Apply the patch
    base_model.BaseModel.__init__ = patched_init
    
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
    compressed_model_path = None
    try:
        # Check if GETA optimizer config exists
        optimizer_cfg = cfgs.get('geta_optimizer_cfg', {})
        if optimizer_cfg:
            print("Found GETA optimizer configuration in config file")
            for k, v in optimizer_cfg.items():
                print(f"  {k}: {v}")
        
        # Use the model's built-in init_geta method if available
        if hasattr(model, 'init_geta'):
            print("Using model's init_geta method")
            model.init_geta(dummy_input, optimizer_cfg)
        elif not hasattr(model, 'oto') or model.oto is None:
            print("Creating new OTO instance")
            model.oto = OTO(model=model, dummy_input=dummy_input)
            # Initialize compression steps
            model.oto.partition_pzigs()
            model.oto.set_trainable()
            
        # For the compression to be effective, manually set target sparsity
        target_group_sparsity = optimizer_cfg.get('target_group_sparsity', 0.5)
        if hasattr(model.oto, '_graph'):
            msg_mgr = get_msg_mgr()
        msg_mgr.log_info(f"Setting target sparsity to {target_group_sparsity}")
        model.oto._graph.random_set_zero_groups(target_group_sparsity=target_group_sparsity)
            
        # Apply compression through training iterations
        msg_mgr.log_info("Applying compression through training iterations...")
        
        # Create dummy criterion for training
        criterion = torch.nn.CrossEntropyLoss()
        
        # Set model to training mode
        model.train()
        
        # Get optimizer configuration
        target_group_sparsity = optimizer_cfg.get('target_group_sparsity', 0.5)
        pruning_steps = optimizer_cfg.get('pruning_steps', 1000)
        
        # Simulate a number of training iterations (at least pruning_steps + some extra)
        num_iterations = max(pruning_steps + 200, 1500)
        msg_mgr.log_info(f"Running {num_iterations} training iterations to apply compression...")
        
        # Track sparsity and metrics
        for i in range(num_iterations):
            # Forward pass with dummy input
            outputs = model(dummy_input)
            
            # Extract logits (assuming model returns a dict with training_feat->softmax->logits)
            logits = outputs['training_feat']['softmax']['logits']
            
            # Generate random target labels for the dummy loss
            batch_size = dummy_input[0].size(0)
            random_labels = torch.randint(0, model_cfg['class_num'], (batch_size,)).cuda()
            
            # Calculate loss
            loss = criterion(logits, random_labels)
            
            # Backward and optimize
            model.geta_optimizer.zero_grad()
            loss.backward()
            model.geta_optimizer.step()
            
            # Track metrics
            if i % 100 == 0 or i == num_iterations - 1:
                # Get metrics from optimizer
                metrics = model.geta_optimizer.compute_metrics()
                current_sparsity = metrics.group_sparsity
                msg_mgr.log_info(f"Iteration {i}/{num_iterations}, "
                     f"Loss: {loss.item():.4f}, "
                     f"Group Sparsity: {current_sparsity:.4f}, "
                     f"Important Groups: {metrics.num_important_groups}, "
                     f"Redundant Groups: {metrics.num_redundant_groups}")
        
        msg_mgr.log_info("Training iterations complete")
        
        # Now construct compressed model
        model.eval()  # Set to evaluation mode
        if hasattr(model, 'construct_compressed_model'):
            msg_mgr.log_info("Using model's construct_compressed_model method")
            compressed_model_path = model.construct_compressed_model(out_dir=args.output_dir)
        else:
            # Alternative approach if method doesn't exist
            msg_mgr.log_info("Using OTO's construct_subnet method directly")
            out_name = f"{model_cfg['model']}_compressed"
            msg_mgr.log_info(f"Saving compressed model as {out_name}")
            model.oto.construct_subnet(
                out_dir=args.output_dir,
                compressed_model_dir=args.output_dir,
                out_name=out_name
            )
            compressed_model_path = model.oto.compressed_model_path
    except Exception as e:
        msg_mgr.log_warning(f"Error initializing OTO or constructing compressed model: {e}")
        msg_mgr.log_warning("Traceback:")
        import traceback
        msg_mgr.log_warning(traceback.format_exc())
        compressed_model_path = None
    
    if compressed_model_path:
        msg_mgr.log_info(f"Compressed model saved to: {compressed_model_path}")
        
        try:
            # For PyTorch 2.6+ with stricter loading security
            print("Loading compressed model for analysis...")
            model_loaded = False
            
            try:
                # Try with weights_only=False to allow custom classes
                compressed_model = torch.load(compressed_model_path, weights_only=False)
                print("Successfully loaded model with weights_only=False")
                model_loaded = True
            except Exception as e:
                print(f"Failed with weights_only=False: {e}")
                try:
                    # Try with pickle_module=None for older PyTorch versions
                    compressed_model = torch.load(compressed_model_path, pickle_module=None)
                    print("Successfully loaded model with pickle_module=None")
                    model_loaded = True
                except Exception as e:
                    print(f"Failed with pickle_module=None: {e}")
                    try:
                        # Final attempt with map_location
                        compressed_model = torch.load(
                            compressed_model_path, 
                            map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                            weights_only=False
                        )
                        model_loaded = True
                    except Exception as e:
                        print(f"All model loading attempts failed: {e}")
                        print("Continuing with file size analysis only")
            
            # Compare file sizes - this can be done even if model loading failed
            original_size_mb = os.path.getsize(args.checkpoint) / (1024*1024)
            compressed_size_mb = os.path.getsize(compressed_model_path) / (1024*1024)
            
            print(f"\nOriginal model file size: {original_size_mb:.2f} MB")
            print(f"Compressed model file size: {compressed_size_mb:.2f} MB")
            print(f"File size reduction: {(1-compressed_size_mb/original_size_mb)*100:.2f}%")
            
            # Only analyze parameters if model was successfully loaded
            if model_loaded:
                # Count parameters
                original_params = sum(p.numel() for p in model.parameters())
                compressed_params = sum(p.numel() for p in compressed_model.parameters())
                
                print(f"\nOriginal model parameters: {original_params:,}")
                print(f"Compressed model parameters: {compressed_params:,}")
                print(f"Compression ratio: {compressed_params/original_params:.4f}")
                print(f"Parameter reduction: {(1-compressed_params/original_params)*100:.2f}%")
        except Exception as e:
            print(f"Error analyzing compressed model: {e}")
    else:
        print("Failed to construct compressed model.")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()