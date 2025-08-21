#!/bin/bash
set -e

# ----------------------------
# High-Performance Image Stitching Installation Script
# ----------------------------

PROJECT_DIR="$(pwd)/HPIS"

echo "Installing required packages..."
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev

echo "Creating build directory..."
mkdir -p "$PROJECT_DIR/build"
cd "$PROJECT_DIR/build"

echo "Running CMake..."
cmake ..

echo "Building the project..."
make -j$(nproc)

echo "Installation and build completed!"
echo "Executable can be found in $PROJECT_DIR/build/app"
echo "To run the application, use: ./app inputfolder"