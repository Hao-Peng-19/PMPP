cmake_minimum_required(VERSION 3.26)

project(QUBOBruteForcing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Try to detect CUDA
find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA FOUND → Building GPU + CPU version")
    set(HAVE_CUDA TRUE)
else()
    message(WARNING "CUDA NOT FOUND → Building CPU-ONLY VERSION")
    set(HAVE_CUDA FALSE)
endif()

if(HAVE_CUDA)
    include_directories("${CUDAToolkit_INCLUDE_DIRS}")
endif()

# Executable
if(HAVE_CUDA)
    enable_language(CUDA)   # only enable CUDA if toolkit found

    add_executable(${PROJECT_NAME}
        cpu_brute_force.h
        cuda_debug.h
        cuda_timer.h
        cuda_util.h
        datatypes.h
        gpu_brute_force.h
        gpu_brute_force.cu
        main.cpp
        matrix.h
        matrix_reader.h
        qubo_brute_forcer.h 
        qubo_energy.h
        state_vector.h
        timer.h
    )

    target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)

else()
    add_executable(${PROJECT_NAME}
        cpu_brute_force.h
        datatypes.h
        main.cpp
        matrix.h
        matrix_reader.h
        qubo_brute_forcer.h 
        qubo_energy.h
        state_vector.h
        timer.h
    )
endif()
