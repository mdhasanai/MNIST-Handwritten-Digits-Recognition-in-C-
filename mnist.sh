#!/bin/bash
echo "Running neural network training and testing..."

# Check if the configuration file is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <input_config>"
    exit 1
fi

CONFIG_FILE=$1

# Check if the configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found."
    exit 1
fi

# Check if the train executable exists
if [ ! -f "./build/train" ]; then
    echo "Error: 'train' executable not found. Please run build.sh first."
    exit 1
fi

# Run the neural network with the provided configuration file
# ./train "$CONFIG_FILE"
./build/train "$CONFIG_FILE"

# Check if the execution was successful
if [ $? -eq 0 ]; then
    echo "Training and testing completed successfully."
else
    echo "Error: Training and testing failed."
    exit 1
fi