cmake_minimum_required(VERSION 3.0)
project(rehline LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(rehline MODULE src/rehline.cpp)

install(TARGETS rehline DESTINATION .)
