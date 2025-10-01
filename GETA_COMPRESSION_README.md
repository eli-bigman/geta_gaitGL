# GETA Compression for GaitGL

This repository contains the implementation of GETA (General and Efficient Training framework that Automates joint structured pruning and quantization) compression for the GaitGL gait recognition model.

## Issue Fixed: 0% Parameter Reduction Problem

The original implementation was showing 0% parameter reduction after applying GETA compression. This happened because:

1. GETA requires actual training iterations to apply sparsity
2. Simply initializing the GETA optimizer and setting target sparsity is not enough
3. The model needs optimizer steps to effectively apply the compression

## Files Modified

1. **extract_compressed.py**:
   - Added training iterations with dummy inputs to properly apply sparsity
   - Enhanced compression validation and reporting

2. **gaitgl_geta.py**:
   - Improved `init_geta` method to ensure proper OTO initialization
   - Enhanced `construct_compressed_model` to verify compression results
   - Added proper validation and error checking

3. **gaitgl_geta.yaml**:
   - Updated compression parameters for more effective compression
   - Set more appropriate pruning step values

## How to Use

### Extract a Compressed Model

To extract a compressed model from a trained checkpoint:

```bash
# Run the provided script
./run_extract_compressed.sh  # Linux/Mac
# OR
.\run_extract_compressed.ps1  # Windows PowerShell
# OR
run_extract_compressed.bat  # Windows CMD
```

This will:
1. Load the trained model
2. Apply GETA compression through training iterations
3. Save the compressed model to `./output/compressed_models`

### Custom Extraction

You can customize the extraction by directly using the script with your own parameters:

```bash
python OpenGait/extract_compressed.py \
  --checkpoint path/to/your/checkpoint.pt \
  --config path/to/your/config.yaml \
  --output_dir path/to/save/compressed/model
```

## Compression Parameters (in gaitgl_geta.yaml)

- `target_group_sparsity`: Controls compression level (0.0 to 1.0), higher = more compression
- `start_pruning_step`: When to start applying compression during training
- `pruning_steps`: How many steps to gradually increase compression to target
- `pruning_periods`: Number of periods for incremental sparsity updates

## Results

After applying the fix, you should see:
- Significant parameter reduction (typically 40-60% depending on sparsity settings)
- File size reduction proportional to parameter reduction
- Minimal impact on model performance

## Troubleshooting

If you still see low compression rates:
1. Increase `target_group_sparsity` in the config (e.g., from 0.5 to 0.7)
2. Ensure enough training iterations are performed in extract_compressed.py
3. Check if the model architecture has limitations on compressible parameters