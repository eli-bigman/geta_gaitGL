# Methodology: Integrating GETA Compression with GaitBase

## 1. Introduction

This methodology documents the integration of GETA (General and Efficient Training frAmework) compression technique with the GaitBase model for efficient gait recognition. The integration leverages the OTO (Only-Train-Once) framework to achieve model compression while maintaining recognition accuracy.

The methodology builds upon the OpenGait framework, which provides a comprehensive platform for gait recognition. This document covers both the baseline implementation of the GaitBase model and the enhanced version with GETA compression.

## 2. Prerequisites and Environment Setup

Before implementing the GETA compression with GaitBase, ensure the following prerequisites are met:

1. **Environment Setup**:
   - Python environment with PyTorch (>=1.10)
   - Required dependencies: torchvision, pyyaml, tensorboard, opencv-python, tqdm, kornia, einops
   - CUDA-compatible GPU(s) for training and inference

   Installation can be done via Anaconda:
   ```bash
   conda install tqdm pyyaml tensorboard opencv kornia einops -c conda-forge
   conda install pytorch>=1.10 torchvision -c pytorch
   ```

2. **Dataset Preparation**:
   - CASIA-B dataset processed into pickle format
   - Directory structure following OpenGait requirements:
     ```
     CASIA-B-pkl
         001 (subject)
             bg-01 (type)
                 000 (view)
                     000.pkl (contains all frames)
                 ...
             ...
         ...
     ```
   - Dataset partitioning file for train/test split

## 3. System Architecture

The implementation modifies the original GaitBase architecture by incorporating GETA compression capabilities, which include both structured pruning and quantization. This section describes the architectural changes made to support model compression.

### 3.1 Implementation File Structure

The implementation involves creating a new Python file in the `opengait/modeling/models/` directory:

```
opengait/
└── modeling/
    └── models/
        ├── baseline.py          # Original GaitBase model
        └── baseline_geta.py     # New file with GETA compression capabilities
```

We created `baseline_geta.py` for the GaitBase model to implement the compression techniques while maintaining the original model files unchanged.

### 3.2 GaitBase Model Architecture

The baseline GaitBase model is a silhouette-based gait recognition architecture with the following key components:
- Silhouette feature extraction through convolutional layers
- Temporal aggregation of frame features
- Horizontal Pyramid Pooling for multi-scale feature representation
- Separate fully connected layers and BN necks for classification

### 3.3 GaitBase with GETA Model Architecture

A new model class `GaitBaseGeta` defined in `baseline_geta.py` extends the base architecture with GETA compression capabilities:

```python
class GaitBaseGeta(BaseModel):
    """
    GaitBase with GETA compression: A silhouette-based gait recognition model with compression capabilities
    """

    def __init__(self, *args, **kargs):
        super(GaitBaseGeta, self).__init__(*args, **kargs)
        self.oto = None
        self.optimizer = None
        self.hesso_optimizer = None
```

The model builds upon the original architecture while adding support for structural pruning. The primary components include:
- Silhouette feature extraction layers
- Horizontal Pyramid Pooling for multi-scale feature representation
- Separate fully connected layers for classification
- GETA compression integration through OTO framework

The baseline model's configuration typically includes:

```yaml
model_cfg:
  model: Baseline
  backbone_cfg:
    in_channels: 1
    layers_cfg: 
      - BC-64
      - BC-64
      - M
      - BC-128
      - BC-128
      - M
      - BC-256
      - BC-256
    type: Plain
  SeparateFCs:
    in_channels: 256
    out_channels: 256
    parts_num: 31
  SeparateBNNecks:
    class_num: 74
    in_channels: 256
    parts_num: 31
  bin_num:
    - 16
    - 8
    - 4
    - 2
    - 1
```

## 4. GETA Compression Integration

### 4.1 OTO Instance Initialization

Two primary methods were added to handle model compression initialization:

```python
def init_hesso(self, dummy_input, optimizer_cfg):
    """Initialize HESSO optimizer within GETA for the model"""
    # Save current training state
    training_state = self.training
    
    # Temporarily set model to eval mode for tracing
    self.eval()
    
    # Initialize OTO instance for GaitBase model
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
```

### 4.2 Structural Optimization

```python
def init_hesso(self, dummy_input, optimizer_cfg):
    """Initialize HESSO optimizer within GETA for the model"""
    # Save current training state
    training_state = self.training
    
    # Temporarily set model to eval mode for tracing
    self.eval()
    
    # Initialize OTO instance for GaitBase model
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
```

### 4.3 Compressed Model Construction

After training, the compressed model is constructed using:

```python
def construct_compressed_model(self, out_dir='./output'):
    """Construct compressed model after training"""
    if self.oto:
        import os
        import torch
        from datetime import datetime
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"compressed_gaitbase_geta_{now}.pt")
        self.oto.construct_subnet(out_dir=os.path.dirname(out_path), 
                                 out_name=os.path.basename(out_path))
        return self.oto.compressed_model_path
    return None
```

### 4.4 Forward Method Adaptation

The forward method was adapted to handle both normal execution and tracing during OTO initialization:

```python
def forward(self, inputs):
    ipts, labs, _, _, seqL = inputs
    seqL = None if not self.training else seqL
    
    # Only check batch size in real inference, not during tracing
    if not self.training and not torch._C._get_tracing_state() and labs.shape[0] != 1:
        raise ValueError(
            'The input size of each GPU must be 1 in testing mode, but got {}!'.format(labs.shape[0]))
    
    # Process silhouettes...
    # [Rest of the forward method implementation]
```

## 5. Main Entry Point Adaptation

A specialized `main_geta.py` script was created to support model training with GETA compression:

```python
def run_model(cfgs, training):
    msg_mgr = get_msg_mgr()
    model_cfg = cfgs['model_cfg']
    msg_mgr.log_info(model_cfg)
    Model = getattr(models, model_cfg['model'])
    model = Model(cfgs, training)
    
    # Create a dummy input for the model - necessary for OTO
    batch_size = 4
    seq_len = 30  # Frames per sequence
    height = 64   # Height of silhouette
    width = 44    # Width of silhouette
    
    # Silhouette images in batch form - use random data but with fixed seed for reproducibility
    torch.manual_seed(42)
    sils = torch.rand(batch_size, seq_len, 1, height, width).cuda()
    # Labels, types, views, seqL
    labs = torch.zeros(batch_size).long().cuda()
    typs = torch.zeros(batch_size).long().cuda()
    vies = torch.zeros(batch_size).long().cuda()
    seqL = torch.full((batch_size,), seq_len).long().cuda()
    
    dummy_input = [sils, labs, typs, vies, seqL]
    
    if training and cfgs['trainer_cfg']['sync_BN']:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    if 'compression_optimizer' in cfgs and training:
        # Initialize the HESSO optimizer
        if cfgs['compression_optimizer'] == 'hesso':
            optimizer = model.init_hesso(dummy_input, cfgs['hesso_optimizer_cfg'])
            msg_mgr.log_info("Using HESSO optimizer for compression")
        else:
            msg_mgr.log_info("No compression optimizer specified, using standard optimizer")
            optimizer = None
    else:
        optimizer = None
    
    # [Rest of the function implementation]
```

## 6. Configuration Changes

OpenGait uses YAML configuration files to define model parameters, data processing, training, and evaluation settings. For the baseline model, the configuration typically includes data settings, model architecture, optimizer parameters, and evaluation metrics.

### 6.1 Standard Configuration

The standard OpenGait configuration includes several sections:

- `data_cfg`: Defines dataset name, paths, and preprocessing parameters
- `model_cfg`: Specifies model architecture and parameters
- `optimizer_cfg`: Sets optimization algorithm and learning rate
- `scheduler_cfg`: Controls learning rate scheduling
- `trainer_cfg`: Defines training process parameters
- `evaluator_cfg`: Sets evaluation metrics and processes

### 6.2 GETA Configuration

A specialized YAML configuration file (`gaitbase_geta.yaml`) was created to support the GETA compression:

```yaml
data_cfg:
  dataset_name: CASIA-B
  dataset_root: /path/to/dataset
  dataset_partition: ./datasets/CASIA-B/CASIA-B.json
  num_workers: 1
  remove_no_gallery: false
  test_dataset_name: CASIA-B

model_cfg:
  model: GaitBaseGeta
  channels: [32, 64, 128]
  class_num: 74

# GETA optimizer configuration
geta_optimizer_cfg:
  variant: adam  # Optimizer type: sgd, adam, or adamw
  lr: 1.0e-4
  weight_decay: 5.0e-4
  target_group_sparsity: 0.5  # Higher sparsity leads to more compression but potentially lower accuracy
  start_pruning_step: 10000  # When to start pruning
  pruning_steps: 20000  # How many steps to reach target sparsity
  pruning_periods: 10  # Number of periods for incremental pruning

# Which optimizer to use
compression_optimizer: geta
```

## 7. Training and Testing Process

### 7.1 Standard Training Process

The standard training process for the GaitBase model uses distributed data parallel (DDP) training:

```powershell
# Training script for standard GaitBase
$env:CUDA_VISIBLE_DEVICES="0,1"
python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs ./configs/gaitbase/gaitbase.yaml --phase train
```

Key training parameters from the standard configuration:
- Batch size: Typically [8, 16] for triplet sampling (8 subjects, 16 sequences per subject)
- Fixed frame count: 30 frames per sequence
- Learning rate: 0.1 with SGD optimizer
- Weight decay: 0.0005
- Milestones for learning rate reduction at steps 20000, 40000

### 7.2 GETA Training and Testing

Specialized scripts were created for training and testing the compressed model:

**Training Script (train_geta.ps1):**
```powershell
# Training script for GaitBase with GETA
$env:CUDA_VISIBLE_DEVICES="0,1"
python -m torch.distributed.launch --nproc_per_node=2 opengait/main_geta.py --cfgs ./configs/gaitbase/gaitbase_geta.yaml --phase train
```

**Testing Script (test_geta.ps1):**
```powershell
# Testing script for GaitBase with GETA
$env:CUDA_VISIBLE_DEVICES="0,1"
python -m torch.distributed.launch --nproc_per_node=2 opengait/main_geta.py --cfgs ./configs/gaitbase/gaitbase_geta.yaml --phase test
```

The training process with HESSO includes both model training and structural pruning as determined by the HESSO optimizer configuration parameters.

## 8. Optimization Process

### 8.1 Standard Training Optimization

The standard GaitBase model is trained with:
- SGD optimizer with momentum (0.9)
- Learning rate of 0.1, reduced at specific milestones
- Cross-entropy and triplet losses for identity discrimination
- Half-precision floating point (when enabled) for memory efficiency

### 8.2 GETA Compression Process

The GETA compression technique involves several stages during model training:

1. **Initialization Phase**: The model is initialized with the OTO framework for structural pruning.

2. **Training Phase**: The model is trained using the GETA framework with the selected optimizer (e.g., HESSO), which gradually introduces sparsity according to the configuration parameters:
   - `start_pruning_step`: When to begin applying sparsity constraints
   - `pruning_steps`: Duration over which to increase sparsity
   - `pruning_periods`: Number of discrete steps for sparsity increments
   - `target_group_sparsity`: Final sparsity level to achieve

3. **Compression Phase**: After training, the `construct_compressed_model` method is called to generate a pruned version of the model with reduced parameter count and computational requirements.

### 8.3 Evaluation Process

The evaluation metrics for both standard and compressed models include:
- Rank-1 identification accuracy
- Euclidean or cosine distance metrics
- Cross-view performance analysis (CASIA-B dataset has 11 views)

## 9. Benefits and Trade-offs

The integration of GETA compression with GaitBase offers several advantages:

1. **Reduced Model Size**: The pruning process significantly reduces the number of parameters in the model.

2. **Lower Computational Requirements**: Structural pruning reduces the computational complexity (FLOPs and MACs).

3. **Minimal Accuracy Degradation**: With proper configuration, the compressed model maintains competitive accuracy compared to the uncompressed version.

4. **Deployment Efficiency**: The compressed model is more suitable for deployment on resource-constrained devices.

The primary trade-off is the additional complexity during the training process and the need for careful tuning of compression hyperparameters to maintain accuracy while achieving the desired level of compression.

## 10. Implementation Guidelines

### 10.1 Installation Steps
1. Install required dependencies as outlined in the prerequisites section
2. Prepare the dataset following the OpenGait format requirements
3. Configure the YAML configuration files for baseline and HESSO versions
4. Execute training and testing scripts for both models

### 10.2 Common Issues and Solutions
- Memory errors: Reduce batch size or use half precision (set `enable_float16: true`)
- DDP zombie processes: Use the provided cleanup script if training terminates abnormally
- Overfitting: Adjust the `target_group_sparsity` value to balance compression and accuracy

## 11. Conclusion

This methodology demonstrates the successful integration of GETA compression technique with the GaitBase model for efficient gait recognition. The approach maintains the recognition performance while significantly reducing the model size and computational requirements through structural pruning, making it more suitable for deployment in resource-constrained environments.

The integration leverages the OpenGait framework's modular design, enabling straightforward extension of the baseline model with advanced compression capabilities. By following this methodology, researchers can implement their own compressed gait recognition models with minimal modifications to the underlying architecture.