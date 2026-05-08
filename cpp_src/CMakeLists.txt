cmake_minimum_required(VERSION 3.10)
project(RT_Intent_Classifier)

set(CMAKE_CXX_STANDARD 17)

find_path(ONNXRUNTIME_INCLUDE_DIR 
    NAMES onnxruntime_cxx_api.h
    PATH_SUFFIXES onnxruntime
)

find_library(ONNXRUNTIME_LIBRARY 
    NAMES onnxruntime
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ONNX Runtime not found! Try to install it with your package manager")
endif()

find_package(nlohmann_json REQUIRED)
find_package(Threads REQUIRED)

include_directories(${ONNXRUNTIME_INCLUDE_DIR})
include_directories(/usr/include/onnxruntime) 

add_executable(rt_inference main.cpp)
target_link_libraries(rt_inference ${ONNXRUNTIME_LIBRARY} nlohmann_json::nlohmann_json Threads::Threads)
