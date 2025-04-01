#!/bin/bash

# Script to read a dataset image and output a tensor

# Display usage information if the incorrect number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <image_dataset_input> <image_tensor_output> <image_index>"
    exit 1
fi

# Assign input arguments to descriptive variables
IMAGE_DATASET_INPUT=$1
IMAGE_TENSOR_OUTPUT=$2
IMAGE_INDEX=$3

# Informing about the operation being performed
echo "Processing dataset image..."
echo "Input: $IMAGE_DATASET_INPUT"
echo "Output: $IMAGE_TENSOR_OUTPUT"
echo "Index: $IMAGE_INDEX"

# Execute the C++ program to process the dataset image
./build/read_dataset_images "$IMAGE_DATASET_INPUT" "$IMAGE_TENSOR_OUTPUT" "$IMAGE_INDEX"

# Confirm completion
if [ $? -eq 0 ]; then
    echo "Operation completed successfully."
else
    echo "An error occurred during processing."
    exit 1
fi
# read_dataset_images.sh mnist-datasets/train-images.idx3-ubyte image_out.txt 0