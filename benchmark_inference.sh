#!/bin/bash

onnx_paths=("model.onnx") 
batch_sizes=(1 4 16 64 256 1024)
devices=("singlethreaded" "cpu" "gpu")
backends=("torch" "onnxruntime" "keras" "sofie")
output="output.csv" 

# Loop over all possible combinations of parameters
for onnx_path in "${onnx_paths[@]}"; do
  for batch_size in "${batch_sizes[@]}"; do
    for device in "${devices[@]}"; do
      for backend in "${backends[@]}"; do
        echo "Running with onnx_path=$onnx_path, batch_size=$batch_size, device=$device, backend=$backend"
        python3 benchmark_inference.py --onnx_path "$onnx_path" --batch_size "$batch_size" --device "$device" --backend "$backend" --output "$output"
      done
    done
  done
done