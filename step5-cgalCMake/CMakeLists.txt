cmake_minimum_required(VERSION 3.1.0)
project(cgalc)

find_package(CGAL REQUIRED)
include(${CGAL_USE_FILE})

add_executable(${PROJECT_NAME} "main.cpp")

target_link_libraries(${PROJECT_NAME} ${CGAL_LIBS})
