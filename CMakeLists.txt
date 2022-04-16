cmake_minimum_required(VERSION 3.21)
project(nazarenko_project)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(nazarenko_program main.cpp)
target_link_libraries(nazarenko_program ${OpenCV_LIBS} tbb)
