#!/bin/bash

if [[ $EUID -ne 0 ]]; then
  echo "Error: Program must be run as root (sudo)"
  exit 1
fi

BUILD_DIR="build"
BINARY="./build/rt_inference"

if [[ ! -f "$BINARY" ]]; then
  echo "Binary not found"
  cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$BUILD_DIR"
  echo "Binary successfully compiled"
fi

taskset -c 3 ./build/rt_inference "$@"
