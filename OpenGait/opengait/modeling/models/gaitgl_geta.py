"""The implementation of GaitGL with GETA compression.
   
This module is modified from the original GaitGL model to support GETA compression technique.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel
from ..modules import SeparateFCs, BasicConv3d, PackSequenceWrapper, SeparateBNNecks
from only_train_once import OTO
from only_train_once.quantization.quant_model import model_to_quantize_model
from only_train_once.quantization.quant_layers import QuantizationMode


class GLConv(nn.Module):
    def __init__(self, in_channels, out_channels, halving, fm_sign=False, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, **kwargs):
        super(GLConv, self).__init__()
        self.halving = halving
        self.fm_sign = fm_sign
        self.global_conv3d = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)
        self.local_conv3d = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)

    def forward(self, x):
        '''
            x: [n, c, s, h, w]
        '''
        gob_feat = self.global_conv3d(x)
        if self.halving == 0:
            lcl_feat = self.local_conv3d(x)
        else:
            h = x.size(3)
            split_size = int(h // 2**self.halving)
            lcl_feat = x.split(split_size, 3)
            lcl_feat = torch.cat([self.local_conv3d(_) for _ in lcl_feat], 3)

        if not self.fm_sign:
            feat = F.leaky_relu(gob_feat) + F.leaky_relu(lcl_feat)
        else:
            feat = F.leaky_relu(torch.cat([gob_feat, lcl_feat], dim=3))
        return feat


class GeMHPP(nn.Module):
    def __init__(self, bin_num=[64], p=6.5, eps=1.0e-6):
        super(GeMHPP, self).__init__()
        self.bin_num = bin_num
        self.p = nn.Parameter(
            torch.ones(1)*p)
        self.eps = eps

    def gem(self, ipts):
        return F.avg_pool2d(ipts.clamp(min=self.eps).pow(self.p), (1, ipts.size(-1))).pow(1. / self.p)

    def forward(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p] 
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = self.gem(z).squeeze(-1)
            features.append(z)
        return torch.cat(features, -1)


class GaitGLGeta(BaseModel):
    """
    GaitGL with GETA compression: Gait Recognition via Effective Global-Local Feature Representation and Local Temporal Aggregation
    """

    def __init__(self, *args, **kargs):
        super(GaitGLGeta, self).__init__(*args, **kargs)
        self.oto = None
        self.geta_optimizer = None

    def build_network(self, model_cfg):
        in_c = model_cfg['channels']
        class_num = model_cfg['class_num']
        dataset_name = self.cfgs['data_cfg']['dataset_name']
        self.quantize = model_cfg.get('quantize', True)
        self.activation_quantize = model_cfg.get('activation_quantize', True)

        if dataset_name in ['OUMVLP', 'GREW']:
            # For OUMVLP and GREW
            self.conv3d = nn.Sequential(
                BasicConv3d(1, in_c[0], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True),
                BasicConv3d(in_c[0], in_c[0], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True),
            )
            self.LTA = nn.Sequential(
                BasicConv3d(in_c[0], in_c[0], kernel_size=(
                    3, 1, 1), stride=(3, 1, 1), padding=(0, 0, 0)),
                nn.LeakyReLU(inplace=True)
            )

            self.GLConvA0 = nn.Sequential(
                GLConv(in_c[0], in_c[1], halving=1, fm_sign=False, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                GLConv(in_c[1], in_c[1], halving=1, fm_sign=False, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            )
            self.MaxPool0 = nn.MaxPool3d(
                kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.GLConvA1 = nn.Sequential(
                GLConv(in_c[1], in_c[2], halving=1, fm_sign=False, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                GLConv(in_c[2], in_c[2], halving=1, fm_sign=False, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            )
            self.GLConvB2 = nn.Sequential(
                GLConv(in_c[2], in_c[3], halving=1, fm_sign=False,  kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                GLConv(in_c[3], in_c[3], halving=1, fm_sign=True,  kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            )
        else:
            # For CASIA-B or other unstated datasets.
            self.conv3d = nn.Sequential(
                BasicConv3d(1, in_c[0], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True)
            )
            self.LTA = nn.Sequential(
                BasicConv3d(in_c[0], in_c[0], kernel_size=(
                    3, 1, 1), stride=(3, 1, 1), padding=(0, 0, 0)),
                nn.LeakyReLU(inplace=True)
            )

            self.GLConvA0 = GLConv(in_c[0], in_c[1], halving=3, fm_sign=False, kernel_size=(
                3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            self.MaxPool0 = nn.MaxPool3d(
                kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.GLConvA1 = GLConv(in_c[1], in_c[2], halving=3, fm_sign=False, kernel_size=(
                3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            self.GLConvB2 = GLConv(in_c[2], in_c[2], halving=3, fm_sign=True,  kernel_size=(
                3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = GeMHPP()

        self.Head0 = SeparateFCs(64, in_c[-1], in_c[-1])

        if 'SeparateBNNecks' in model_cfg.keys():
            self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
            self.Bn_head = False
        else:
            self.Bn = nn.BatchNorm1d(in_c[-1])
            self.Head1 = SeparateFCs(64, in_c[-1], class_num)
            self.Bn_head = True

        # Apply quantization if configured
        if self.quantize:
            quant_mode = QuantizationMode.WEIGHT_AND_ACTIVATION if self.activation_quantize else QuantizationMode.WEIGHT_ONLY
            self = model_to_quantize_model(self, quant_mode=quant_mode)

    def init_geta(self, dummy_input, optimizer_cfg):
        """Initialize GETA optimization for the model"""
        # Save current training state
        training_state = self.training
        
        # Temporarily set model to eval mode for tracing
        self.eval()
        
        # Initialize OTO instance for GaitGL model
        try:
            # Only create new OTO instance if one doesn't exist
            if not hasattr(self, 'oto') or self.oto is None:
                self.oto = OTO(model=self, dummy_input=dummy_input)
                
            # Make sure we partition the graph for proper compression
            if hasattr(self.oto, 'partition_pzigs'):
                self.oto.partition_pzigs()
                
            # Make sure trainable parameters are set
            if hasattr(self.oto, 'set_trainable'):
                self.oto.set_trainable()
        finally:
            # Restore original training state
            if training_state:
                self.train()
            else:
                self.eval()
        
        # Create GETA optimizer based on configuration
        self.geta_optimizer = self.oto.geta(
            variant=optimizer_cfg.get('variant', 'adam'),
            lr=optimizer_cfg.get('lr', 1.0e-4),
            lr_quant=optimizer_cfg.get('lr_quant', 1.0e-3),
            first_momentum=optimizer_cfg.get('first_momentum', 0.9),
            weight_decay=optimizer_cfg.get('weight_decay', 5.0e-4),
            target_group_sparsity=optimizer_cfg.get('target_group_sparsity', 0.5),
            start_pruning_step=optimizer_cfg.get('start_pruning_step', 10000),
            pruning_steps=optimizer_cfg.get('pruning_steps', 20000),
            pruning_periods=optimizer_cfg.get('pruning_periods', 10),
        )
        
        # Verify that the optimizer is properly set up
        if hasattr(self.geta_optimizer, 'compute_metrics'):
            metrics = self.geta_optimizer.compute_metrics()
            print(f"Initial group sparsity: {metrics.group_sparsity:.4f}")
            print(f"Target group sparsity: {optimizer_cfg.get('target_group_sparsity', 0.5):.4f}")
            print(f"Important groups: {metrics.num_important_groups}")
            print(f"Redundant groups: {metrics.num_redundant_groups}")
        
        return self.geta_optimizer

    def init_hesso(self, dummy_input, optimizer_cfg):
        """Initialize HESSO optimization for the model"""
        # Save current training state
        training_state = self.training
        
        # Temporarily set model to eval mode for tracing
        self.eval()
        
        # Initialize OTO instance for GaitGL model
        try:
            self.oto = OTO(model=self, dummy_input=dummy_input)
        finally:
            # Restore original training state
            if training_state:
                self.train()
            else:
                self.eval()
        
        # Create HESSO optimizer based on configuration
        self.hesso_optimizer = self.oto.hesso(
            variant=optimizer_cfg.get('variant', 'adam'),
            lr=optimizer_cfg.get('lr', 1.0e-4),
            weight_decay=optimizer_cfg.get('weight_decay', 5.0e-4),
            target_group_sparsity=optimizer_cfg.get('target_group_sparsity', 0.5),
            start_pruning_step=optimizer_cfg.get('start_pruning_step', 10000),
            pruning_steps=optimizer_cfg.get('pruning_steps', 20000),
            pruning_periods=optimizer_cfg.get('pruning_periods', 10),
        )
        
        return self.hesso_optimizer

    def construct_compressed_model(self, out_dir='./output'):
        """Construct compressed model after training"""
        if self.oto:
            # Ensure we're in eval mode for subnet construction
            training_state = self.training
            self.eval()
            
            try:
                # Verify that compression has been applied
                if hasattr(self.geta_optimizer, 'compute_metrics'):
                    metrics = self.geta_optimizer.compute_metrics()
                    print(f"Final group sparsity before construction: {metrics.group_sparsity:.4f}")
                    print(f"Important groups: {metrics.num_important_groups}")
                    print(f"Redundant groups: {metrics.num_redundant_groups}")
                    
                    if metrics.group_sparsity < 0.01:
                        print("WARNING: Group sparsity is too low. GETA compression may not have been properly applied.")
                        print("Did you run training iterations to apply sparsity?")
                
                # Construct the compressed subnet
                self.oto.construct_subnet(out_dir=out_dir)
                
                # Verify compression results
                if hasattr(self.oto, 'compressed_model_path') and hasattr(self.oto, 'full_group_sparse_model_path'):
                    original_size_mb = os.path.getsize(self.oto.full_group_sparse_model_path) / (1024*1024)
                    compressed_size_mb = os.path.getsize(self.oto.compressed_model_path) / (1024*1024)
                    
                    print(f"Full model size: {original_size_mb:.2f} MB")
                    print(f"Compressed model size: {compressed_size_mb:.2f} MB")
                    print(f"Size reduction: {(1-compressed_size_mb/original_size_mb)*100:.2f}%")
                    
                return self.oto.compressed_model_path
            finally:
                # Restore original training state
                if training_state:
                    self.train()
        return None

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        seqL = None if not self.training else seqL
        
        # Only check batch size in real inference, not during tracing
        if not self.training and not torch._C._get_tracing_state() and labs.shape[0] != 1:
            raise ValueError(
                'The input size of each GPU must be 1 in testing mode, but got {}!'.format(labs.shape[0]))
                
        sils = ipts[0].unsqueeze(1)
        del ipts
        n, _, s, h, w = sils.size()
        if s < 3:
            repeat = 3 if s == 1 else 2
            sils = sils.repeat(1, 1, repeat, 1, 1)

        outs = self.conv3d(sils)
        outs = self.LTA(outs)

        outs = self.GLConvA0(outs)
        outs = self.MaxPool0(outs)

        outs = self.GLConvA1(outs)
        outs = self.GLConvB2(outs)  # [n, c, s, h, w]

        outs = self.TP(outs, seqL=seqL, options={"dim": 2})[0]  # [n, c, h, w]
        outs = self.HPP(outs)  # [n, c, p]

        gait = self.Head0(outs)  # [n, c, p]

        if self.Bn_head:  # Original GaitGL Head
            bnft = self.Bn(gait)  # [n, c, p]
            logi = self.Head1(bnft)  # [n, c, p]
            embed = bnft
        else:  # BNNechk as Head
            bnft, logi = self.BNNecks(gait)  # [n, c, p]
            embed = gait

        n, _, s, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed, 'labels': labs},
                'softmax': {'logits': logi, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils.view(n*s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval