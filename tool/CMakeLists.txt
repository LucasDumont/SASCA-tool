cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

project(SACA)

include_directories(.)

add_executable(${PROJECT_NAME}
  "construction.cpp"
  "main.cpp")

FILE(COPY ../sample/example DESTINATION ${CMAKE_BINARY_DIR})
FILE(COPY ../sample/example_exe DESTINATION ${CMAKE_BINARY_DIR})

target_link_libraries(${PROJECT_NAME} hdf5)
