#!/bin/bash
# This script demonstrates how to extract a compressed model from a checkpoint

# Navigate to the OpenGait directory
cd OpenGait

# Run the extraction script
python extract_compressed.py --checkpoint output/CASIA-B/GaitGLGeta/GaitGL_GETA/checkpoints/GaitGL_GETA-80000.pt --config configs/gaitgl/gaitgl_geta.yaml --output_dir ./output/compressed_models