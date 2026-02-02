cmake_minimum_required(VERSION 3.25)
project(EDEN CUDA CXX)

include(FetchContent)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES "120a")

include_directories(csrc)

FetchContent_Declare(
        nanobind
        QUIET
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG v2.9.2
)
find_package(CUDAToolkit REQUIRED)

find_package(Python COMPONENTS Interpreter Development.Module OPTIONAL_COMPONENTS Development.SABIModule)
FetchContent_MakeAvailable(nanobind)
nanobind_add_module(_quartet2 STABLE_ABI
        csrc/binding.cpp
        csrc/group_transform_and_eden.cu
        csrc/group_transform.cu
        csrc/round_four_six.cu
        csrc/round_eden_fp4.cu
        csrc/dq_tp_q.cu
)
target_compile_options(_quartet2 PUBLIC -lineinfo -ffast-math)
target_link_libraries(_quartet2 PUBLIC CUDA::cudart)

# make sure we pick up cuda libraries from within the current python env, if available
set(RPATH_LIST
        "$ORIGIN"
        "$ORIGIN/../nvidia/nccl/lib"
        "$ORIGIN/../nvidia/cudnn/lib"
        "$ORIGIN/../nvidia/cuda_runtime/lib"
        "$ORIGIN/../nvidia/cublas/lib"
)
list(JOIN RPATH_LIST ":" WHEEL_RPATH)

set_target_properties(_quartet2 PROPERTIES
        INSTALL_RPATH "${WHEEL_RPATH}"
        BUILD_WITH_INSTALL_RPATH OFF
        INSTALL_RPATH_USE_LINK_PATH FALSE
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/quartet2
        DESTINATION .
)
install(TARGETS _quartet2 LIBRARY DESTINATION quartet2)
