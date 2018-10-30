cmake_minimum_required(VERSION 2.8)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

project(merge-pointer)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(merge-pointer merge-pointer.cpp)
target_link_libraries(merge-pointer ${OpenCV_LIBS})