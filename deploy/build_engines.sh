#!/bin/bash
# Convert ONNX to TensorRT FP16 engines on the Jetson.
# Run: bash build_engines.sh
set -e
export PATH=/usr/src/tensorrt/bin:$PATH

cd ~/fusion_deploy
mkdir -p build

echo "=== building cam engine (FP16) ==="
trtexec --onnx=onnx/late_2d_cam.onnx --saveEngine=build/late_2d_cam.engine --fp16

echo "=== building lid engine (FP16) ==="
trtexec --onnx=onnx/late_2d_lid.onnx --saveEngine=build/late_2d_lid.engine --fp16

echo "=== done ==="
ls -lh build/*.engine