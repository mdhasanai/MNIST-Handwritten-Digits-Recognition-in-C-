# !/bin/bash
echo "==============================="
echo "       Building Project        "
echo "==============================="
echo "   Team: ws2024-group-44-p36   "
echo "==============================="

# Show current directory
echo "Current directory: $(pwd)"

# Clean up previous build
echo "Cleaning up previous build..."
rm -rf build
mkdir -p build

# Compiler settings
CXX=g++
CXXFLAGS="-Wall -Wextra -pedantic -Werror -Wno-unused-parameter -Wno-unused-variable -g -std=c++20"
OPTIM_FLAGS="-O3"
SRC_DIR="src"
BUILD_DIR="build"
INCLUDE_DIR="include"

# Compile read_dataset_images
echo "Compiling read_dataset_images..."
$CXX $CXXFLAGS -I$INCLUDE_DIR $SRC_DIR/read_dataset_images.cpp -o $BUILD_DIR/read_dataset_images || { echo "Compilation failed!"; exit 1; }

# Compile read_dataset_labels
echo "Compiling read_dataset_labels..."
$CXX $CXXFLAGS -I$INCLUDE_DIR $SRC_DIR/read_dataset_labels.cpp -o $BUILD_DIR/read_dataset_labels || { echo "Compilation failed!"; exit 1; }

# Compile train (with optimization)
echo "Compiling train..."
$CXX $CXXFLAGS $OPTIM_FLAGS -I$INCLUDE_DIR $SRC_DIR/train.cpp -o $BUILD_DIR/train || { echo "Compilation failed!"; exit 1; }

echo "==============================="
echo "      Build Completed!         "
echo "==============================="
echo "   Team: ws2024-group-44-p36   "
echo "==============================="

