
                                VoxelDeformer
==============================================================================

    It is a fast library to deform voxel cubes embedded in tetrahedral meshes.


                              HOW TO BUILD IT?
==============================================================================

        Required Knowledge:
    ===========================

        We use git as source code management tool, in order to become familiar
        with git, you can find good documentation at:

                http://www-cs-students.stanford.edu/~blynn/gitmagic/

        For building our library, we use CMAKE, version 2.8 or later, so it is
        neccesary to have it installed.



        Dependencies:
    =====================

        For developing VoxelDeformer

        Some parts of VoxelDeformer heavily rely on CUDA 5.0 SDK, you can
        download the CUDA SDK installer at:

                    https://developer.nvidia.com/cuda-downloads

        For volume rendering, our demos use OpenGL, GLSL and GLUT, so GLUT
        headers and libraries need to be installed in your system. Fortunatelly,
        these libraries are already present in most modern operative systems.

        In addition, our library uses OpenMP, which is currently included in
        the most recent compilers.




            Building/Running VoxelDeformer in Ubuntu/Linux systems:
==============================================================================

    1) For installing the required libraries, just copy the next line and
       paste it in a shell:

            sudo apt-get install build-essential git cmake freeglut3-dev


    2) Go to https://developer.nvidia.com/cuda-downloads and download the CUDA
       SDK 5.0 or later. Install this sdk following the instructions from the
       web page.


    3) For the development of this library, we use QTCreator, this IDE is
       multiplatform and freely available at:

                        http://qt-project.org/downloads


    4) After installing QtCreator, open it, select "Open project", look for the
       directory where VoxelDeformer was downloaded and select the file
       "CMakeLists.txt".
       After that, QtCreator will ask you for a new directory where to store
       the binaries.
       After creating this directory push "next" button some more times and
       finally VoxelDeformer project is sucessfully loaded.
       Finally, you can build/run the library and demos pressing the "Play"
       button (it is a button situated in the left down corner of the qtCreator
       IDE).


    5) Enjoy and tell us if VoxelDeformer is useful for you! ;-)




              Building/Running VoxelDeformer in Windows Systems:
==============================================================================

TODO: To be completed



              Building/Running VoxelDeformer in Mac OS Systems:
==============================================================================

TODO: To be completed

