#!/bin/bash

# Run octree resolution tests in order
echo "Starting octree resolution tests..."

echo "================================"
echo "Running: 10-6-0.2.yaml"
echo "================================"
python oren/trainer.py --config configs/test-octree-resolution/10-6-0.2.yaml
if [ $? -ne 0 ]; then
    echo "Error: 10-6-0.2.yaml failed"
    exit 1
fi

echo "================================"
echo "Running: 11-7-0.1.yaml"
echo "================================"
python oren/trainer.py --config configs/test-octree-resolution/11-7-0.1.yaml
if [ $? -ne 0 ]; then
    echo "Error: 11-7-0.1.yaml failed"
    exit 1
fi

echo "================================"
echo "Running: 8-5-0.7.yaml"
echo "================================"
python oren/trainer.py --config configs/test-octree-resolution/8-5-0.7.yaml
if [ $? -ne 0 ]; then
    echo "Error: 8-5-0.7.yaml failed"
    exit 1
fi

echo "================================"
echo "Running: 9-6-0.4.yaml"
echo "================================"
python oren/trainer.py --config configs/test-octree-resolution/9-6-0.4.yaml
if [ $? -ne 0 ]; then
    echo "Error: 9-6-0.4.yaml failed"
    exit 1
fi

echo "================================"
echo "All tests completed!"
echo "================================"
