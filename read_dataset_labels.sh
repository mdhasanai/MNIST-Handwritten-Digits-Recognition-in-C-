#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 <label_dataset_input> <label_tensor_output> <label_index>"
    echo "  <label_dataset_input>   Path to the input dataset label file"
    echo "  <label_tensor_output>   Path to the output tensor file"
    echo "  <label_index>           Index of the label to process"
    exit 1
}

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Error: Invalid number of arguments."
    usage
fi

# Assign input arguments to variables
LABEL_DATASET_INPUT=$1
LABEL_TENSOR_OUTPUT=$2
LABEL_INDEX=$3

# Display the input arguments for clarity
echo "Input Dataset Label File: $LABEL_DATASET_INPUT"
echo "Output Tensor File: $LABEL_TENSOR_OUTPUT"
echo "Label Index: $LABEL_INDEX"

# Execute the C++ program to read the dataset label and output the tensor
if ./build/read_dataset_labels "$LABEL_DATASET_INPUT" "$LABEL_TENSOR_OUTPUT" "$LABEL_INDEX"; then
    echo "Successfully processed the dataset label."
else
    echo "Error: Failed to process the dataset label."
    exit 1
fi
# read_dataset_labels.sh mnist-datasets/train-labels.idx1-ubyte label_out.txt 0
