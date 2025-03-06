cmake_minimum_required(VERSION 3.10)

# Set the project name
project(stereo_visual_odometry)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find Ceres
find_package(Ceres REQUIRED)

# Find Abseil explicitly
find_package(absl CONFIG REQUIRED)

# Find GTest (even if you're not using it, resolving dependencies)
find_package(GTest CONFIG REQUIRED)

# Add executable
add_executable(main src/main.cpp)

# Link OpenCV, Ceres, Abseil, and GTest
target_link_libraries(main PRIVATE ${OpenCV_LIBS} Ceres::ceres absl::base absl::strings GTest::gmock stdc++fs)

# Check compiler version and link stdc++fs if necessary
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(main PRIVATE stdc++fs)
endif()
