#!/bin/bash
# Script to extract compressed model from a GETA checkpoint
# Usage: ./extract_compressed.sh --checkpoint /path/to/checkpoint.pt --config /path/to/config.yaml [--output_dir /path/to/save]

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Run the extraction script with passed arguments
python ${SCRIPT_DIR}/extract_compressed.py "$@"

# Print usage if no arguments provided
if [ $# -eq 0 ]; then
  echo "Usage: ./extract_compressed.sh --checkpoint /path/to/checkpoint.pt [options]"
  echo ""
  echo "Required arguments:"
  echo "  --checkpoint PATH      Path to the checkpoint file"
  echo ""
  echo "Optional arguments:"
  echo "  --output-dir DIR       Directory to save the compressed model (default: ./compressed_models)"
  echo "  --model-name NAME      Model class name (default: GaitGLGeta)"
  echo "  --cfg PATH             Path to the model config file (default: ./configs/gaitgl/gaitgl_geta.yaml)"
  echo "  --visualize            Print detailed model statistics comparison"
fi