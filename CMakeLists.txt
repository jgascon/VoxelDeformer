PROJECT (VoxelDeformer)

cmake_minimum_required(VERSION 2.8)
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

# We can force debug or release build uncomenting one of these lines.
#SET(CMAKE_BUILD_TYPE Release)
#SET(CMAKE_BUILD_TYPE Debug)

# OpenGL and Glut are required.
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )

IF(APPLE)
  SET(EXECUTABLE_FLAG MACOSX_BUNDLE)
ENDIF(APPLE)

# Configure warnings in the compiler
if(MSVC)
  # Force to always compile with W4
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
  endif()
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  # Update if necessary
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long")
endif()

find_package(OpenGL REQUIRED)

IF(UNIX)
  IF(CMAKE_COMPILER_IS_GNUCC)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -fpermissive")
    SET(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -Wall -fpermissive")
  ENDIF(CMAKE_COMPILER_IS_GNUCC)
ENDIF(UNIX)

include_directories ("${PROJECT_SOURCE_DIR}")
include_directories ("${PROJECT_SOURCE_DIR}/src")

# Search for OpenMP
FIND_PACKAGE(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} ${OpenMP_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


# Search for CUDA
FIND_PACKAGE(CUDA)
if(CUDA_FOUND)
    SET(CUDA_NVCC_FLAGS "-arch=sm_20;")
    #SET(CUDA_NVCC_FLAGS "-arch;sm_13;-g;-G")
    LINK_LIBRARIES("-L/${CUDA_SDK_ROOT_DIR}/C/lib")

    #Include the FindCUDA script
    INCLUDE(FindCUDA)

else(CUDA_FOUND)
    message("CUDA is not installed on this system.")
endif()



SET (SOURCES
    src/voxelDeformer.hpp
    src/voxelDeformer.cpp
    src/voxelVolume.hpp
    src/voxelVolume.cpp
    src/types.hpp
    src/types.cpp
    src/matrix.hpp
    src/matrix.cpp
    src/shaders.hpp
    src/shaders.cpp
    src/opengl_draw_methods.hpp
    src/opengl_draw_methods.cpp
)


if(CUDA_FOUND)
  CUDA_ADD_EXECUTABLE(simpleDemo ${EXECUTABLE_FLAG} simpleDemo/main.cpp src/culler_rasterizer.cuh src/culler_rasterizer.cu ${SOURCES})
  TARGET_LINK_LIBRARIES(simpleDemo ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})

  add_custom_command(TARGET simpleDemo POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     "${CMAKE_SOURCE_DIR}/simpleDemo/simpleDemo_ray_cast.vert"
                     "${CMAKE_CURRENT_BINARY_DIR}")

  add_custom_command(TARGET simpleDemo POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     "${CMAKE_SOURCE_DIR}/simpleDemo/simpleDemo_ray_cast_3D.frag"
                     "${CMAKE_CURRENT_BINARY_DIR}")

  add_custom_command(TARGET simpleDemo POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     "${CMAKE_SOURCE_DIR}/simpleDemo/Bucky_32x32x32x1.raw"
                     "${CMAKE_CURRENT_BINARY_DIR}")

else(CUDA_FOUND)
    message("CUDA not found. CUDA demos have not been built.")
endif()

