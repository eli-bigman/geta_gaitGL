@echo off
REM This batch file demonstrates how to extract a compressed model from a checkpoint

REM Navigate to the OpenGait directory
cd OpenGait

REM Run the extraction script
python extract_compressed.py --checkpoint output/CASIA-B/GaitGLGeta/GaitGL_GETA/checkpoints/GaitGL_GETA-80000.pt --config configs/gaitgl/gaitgl_geta.yaml --output_dir ./output/compressed_models

REM Pause to see the output
pause