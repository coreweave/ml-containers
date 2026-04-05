# Manually replicate the sccache-friendly settings that PyTorch applies when it
# auto-detects sccache at configure time. Without sccache on PATH during the
# CMake configure step (e.g. when sccache is set manually as a compiler
# launcher), the auto-detection doesn't fire and nvcc falls back to response
# files, which cause sccache to consider the call non-cacheable.
#
# Upstream reference:
#   https://github.com/pytorch/pytorch/blob/v2.11.0/cmake/Dependencies.cmake#L68-L74

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES  OFF CACHE BOOL "Disable response files for includes"  FORCE)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF CACHE BOOL "Disable response files for libraries" FORCE)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS   OFF CACHE BOOL "Disable response files for objects"   FORCE)
