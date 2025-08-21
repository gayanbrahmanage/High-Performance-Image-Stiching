cmake_minimum_required(VERSION 3.10)
project(PanoramaProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/opencv_local/lib/cmake/opencv4")

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# Add include directory for project headers
include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)

add_executable(app src/app.cpp
                   src/ImageStitcher.cpp)

target_link_libraries(app ${OpenCV_LIBS})
