#!/bin/bash
# F1TENTH URDF to USD Conversion Script
# Usage: Run this inside the Isaac Lab Docker container

echo "================================================"
echo "F1TENTH URDF to USD Converter"
echo "================================================"

# Paths (inside container)
URDF_PATH="/workspace/isaaclab/source/isaaclab_assets/isaaclab_assets/f1tenth/f1tenth.urdf"
USD_OUTPUT="/workspace/isaaclab/source/isaaclab_assets/isaaclab_assets/f1tenth/f1tenth.usd"

# Check if URDF exists
if [ ! -f "$URDF_PATH" ]; then
    echo "ERROR: URDF file not found at $URDF_PATH"
    exit 1
fi

echo "Input URDF: $URDF_PATH"
echo "Output USD: $USD_OUTPUT"
echo ""

# Convert URDF to USD
# --merge-joints: REMOVED to preserve sensor mount points (laser link)
# --joint-target-type velocity: Motor control mode
# --joint-damping: Reduced from 0.5 to 0.01 for better velocity tracking
./isaaclab.sh -p scripts/tools/convert_urdf.py \
    "$URDF_PATH" \
    "$USD_OUTPUT" \
    --joint-stiffness 0.0 \
    --joint-damping 0.01 \
    --joint-target-type velocity \
    --headless

echo ""
echo "================================================"
echo "Conversion Complete!"
echo "USD File: $USD_OUTPUT"
echo "================================================"