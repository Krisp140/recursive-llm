#!/bin/bash
# Download LoCoDiff dataset
# This script clones the LoCoDiff-bench repository and copies the dataset

set -e  # Exit on error

echo "============================================================"
echo "LoCoDiff Dataset Download Script"
echo "============================================================"
echo ""

# Get the project root (parent of scripts directory)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Create directories
EXTERNAL_DIR="$PROJECT_ROOT/../external"
DATA_DIR="$PROJECT_ROOT/locodiff_data"

mkdir -p "$EXTERNAL_DIR"
mkdir -p "$DATA_DIR"

# Check if dataset already exists
if [ -d "$DATA_DIR/locodiff-250425/prompts" ]; then
    echo "✓ Dataset already exists at $DATA_DIR/locodiff-250425"
    echo ""
    echo "To re-download, first remove the existing dataset:"
    echo "  rm -rf $DATA_DIR/locodiff-250425"
    exit 0
fi

# Clone LoCoDiff-bench if not already cloned
if [ ! -d "$EXTERNAL_DIR/LoCoDiff-bench" ]; then
    echo "Cloning LoCoDiff-bench repository..."
    cd "$EXTERNAL_DIR"
    git clone https://github.com/AbanteAI/LoCoDiff-bench.git
    echo "✓ Repository cloned"
    echo ""
else
    echo "✓ LoCoDiff-bench repository already exists"
    echo ""
fi

# Copy dataset
echo "Copying dataset to $DATA_DIR..."
cp -r "$EXTERNAL_DIR/LoCoDiff-bench/locodiff-250425" "$DATA_DIR/"
echo "✓ Dataset copied"
echo ""

# Print statistics
PROMPT_COUNT=$(ls "$DATA_DIR/locodiff-250425/prompts/"*_prompt.txt 2>/dev/null | wc -l | tr -d ' ')
EXPECTED_COUNT=$(ls "$DATA_DIR/locodiff-250425/prompts/"*_expectedoutput.txt 2>/dev/null | wc -l | tr -d ' ')

echo "Dataset statistics:"
echo "  Location: $DATA_DIR/locodiff-250425"
echo "  Prompts: $PROMPT_COUNT files"
echo "  Expected outputs: $EXPECTED_COUNT files"
echo ""

# Get size
SIZE=$(du -sh "$DATA_DIR/locodiff-250425" | cut -f1)
echo "  Size: $SIZE"
echo ""

echo "============================================================"
echo "✓ Dataset download complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Verify setup: python locodiff/locodiff_eval/test_setup.py"
echo "  2. Quick test: python locodiff/locodiff_eval/baseline_test.py"
echo "  3. Full evaluation: python locodiff/locodiff_eval/locodiff_evaluation.py"
echo ""
