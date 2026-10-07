cmake_minimum_required(VERSION 3.10)
project(compiler CXX)
find_package(Threads REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")
file(GLOB_RECURSE SRC "src/*.cpp")
include_directories(src)
include_directories(3rd_party/antlr4-runtime)
add_subdirectory(3rd_party/antlr4-runtime)
if(WIN32)
  target_compile_definitions(antlr4_runtime PUBLIC ANTLR4CPP_STATIC)
endif()
add_executable(compiler ${SRC})
target_link_libraries(compiler antlr4_runtime Threads::Threads)

option(SAKURA_BUILD_TESTS "Build compiler regression tests" OFF)
if(SAKURA_BUILD_TESTS)
  enable_testing()
  add_executable(backend_regression tests/backend_regression.cpp
    src/backend/Machine.cpp src/backend/MachineOpt.cpp src/backend/RA.cpp
    src/backend/Asm.cpp src/midend/ir/IR.cpp)
  add_test(NAME backend_regression COMMAND backend_regression)
  add_executable(indvar_regression tests/indvar_regression.cpp
    src/midend/ir/IR.cpp src/midend/pass/OptUtil.cpp src/midend/pass/OptIndVar.cpp
    src/midend/pass/OptIfConv.cpp)
  add_test(NAME indvar_regression COMMAND indvar_regression)
  add_executable(isel_regression tests/isel_regression.cpp
    src/backend/ISel.cpp src/backend/Machine.cpp src/midend/ir/IR.cpp
    src/midend/pass/Passes.cpp src/midend/pass/TensorLower.cpp
    src/midend/pass/OptInline.cpp src/midend/pass/OptUtil.cpp
    src/midend/pass/OptPromote.cpp)
  add_test(NAME isel_regression COMMAND isel_regression)
  add_executable(polyhedral_layout_regression tests/polyhedral_layout_regression.cpp
    src/midend/ir/IR.cpp src/midend/pass/OptUtil.cpp
    src/midend/pass/OptPolyhedral.cpp src/midend/pass/OptBlockLayout.cpp)
  add_test(NAME polyhedral_layout_regression COMMAND polyhedral_layout_regression)
  add_executable(loop_split_regression tests/loop_split_regression.cpp
    src/midend/ir/IR.cpp src/midend/pass/OptUtil.cpp
    src/midend/pass/OptLoopSplit.cpp src/midend/pass/OptLoop.cpp
    src/midend/pass/OptEffects.cpp)
  add_test(NAME loop_split_regression COMMAND loop_split_regression)
  add_executable(memory_regression tests/memory_regression.cpp
    src/midend/ir/IR.cpp src/midend/pass/OptUtil.cpp src/midend/pass/OptMem.cpp
    src/midend/pass/OptEffects.cpp)
  add_test(NAME memory_regression COMMAND memory_regression)
endif()
