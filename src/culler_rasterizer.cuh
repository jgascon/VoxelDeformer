#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector_functions.h>
#include <types.hpp>

#ifndef RASTERIZE_H
#define RASTERIZE_H


#define THREADS_PER_BLOCK 512


void loadCudaInputTexture3D(const VOXELTYPE * h_volume,
                       const unsigned width,
                       const unsigned height,
                       const unsigned depth);


void unLoadCudaInputTexture3D();


void createCudaOutputTexture3D(const unsigned width,
                               const unsigned height,
                               const unsigned depth);


void destroyCudaOutputTexture3D();


///This class stores all the information for each tetrahedron/cell:
typedef struct tetrahedron_info {
    //Original tetrahedron barycentric matrix.
    float bar_mat[9];
    float3 node_0;
    int3 dims;
    int3 iaabb_min;
} tetrahedron_info;


void update_tetrahedra_info_from_tetramesh(vector<tetrahedron_info> & result,
                                           const vector<uvec4> &tetras,
                                           const vector<vec3> &nodes,
                                           bool invert_matrices = false);



///This class stores all the information for each tetrahedron/cell:
typedef struct aabb_info {
    int3 iaabb_min;
    //int3 dims;
    unsigned start_block_offset;
    unsigned index;
    ///Get the corner number "i".
    vec3 get_ith_corner(const unsigned i, const uvec3 aabb_dims = uvec3(8,8,8)) const;
    //Used for performance measuring.
    unsigned grid_box_i;
} aabb_info;




void update_aabb_info_from_tetra_info(vector<aabb_info> & result,
                                      const vector<tetrahedron_info> & tetras,
                                      const uvec3 aabb_dims,
                                      const unsigned threads_per_block = 512);


//-------------------- Computation of masks ---------------


//This grid is used to perform a more efficient culling.
typedef struct culling_grid_node {
    ivec3 node;
    unsigned tetra_i;
} culling_grid_node;




void create_ibox_labeled_grid(iaabb bb,
                              uvec3 new_aabbs_dims,
                              vector<culling_grid_node> & result_nodes,
                              unsigned start_grid_nodes_index,
                              vector<uvec8> & result_boxes,
                              unsigned start_grid_boxes_index,
                              unsigned node_label);


__global__ void cuda_compute_nodes_masks(const unsigned num_grid_nodes,
                                         const tetrahedron_info * tetra_info_1,
                                         const culling_grid_node * grid_nodes,
                                         unsigned char * result_nodes_masks);



void cuda_compute_nodes_masks_call(const unsigned num_grid_nodes,
                                   const tetrahedron_info * tetra_info_1,
                                   const culling_grid_node * grid_nodes,
                                   unsigned char * result_nodes_masks);




__global__ void cuda_compute_boxes_masks(const unsigned num_grid_boxes,
                                         const unsigned char * grid_nodes_masks,
                                         unsigned * grid_boxes,
                                         unsigned char * result_boxes_masks);



void cuda_compute_boxes_masks_call(const unsigned num_grid_boxes,
                                   const unsigned char * grid_nodes_masks,
                                   unsigned * grid_boxes,
                                   unsigned char * result_boxes_masks);



//-------------------- Rasterizer methods ---------------

void cuda_upload_tetrahedra_info(const vector<tetrahedron_info> & tetra_info_1,
                                 tetrahedron_info ** d_tetra_info_1);


void cuda_clear_tetrahedra_info(tetrahedron_info * d_tetra_info_1);


__global__ void cuda_rasterize3D(const tetrahedron_info * tetra_info_0, //Info of each original tetrahedron (aabb, barycentric matrix, etc)
                                 const unsigned * indexes,              //Stores, for each cuda block, its corresponding aabb_info index.
                                 VOXELTYPE * cube_1,                    //Deformed voxel cube [OUT].
                                 const int3 cube_dims_1,                //Dimensions of the deformed cube (width, height, depth).
                                 const tetrahedron_info * tetra_info_1, //Info of each deformed tetrahedron (aabb, barycentric matrix, etc)
                                 const aabb_info * aabb_info_1,         //Info of each aabb (aabb, dims, inferior corner, etc)
                                 const unsigned batch_start_block_offset); //Start block offset of this batch execution.


void cuda_rasterize3DCall(const unsigned num_blocks,
                          const tetrahedron_info * tetra_info_0,
                          const unsigned * indexes,
                          VOXELTYPE * cube_1,
                          const int3 cube_dims_1,
                          const tetrahedron_info * tetra_info_1,
                          const aabb_info * aabb_info_1);


__global__ void cuda_rasterize_tex3D(const tetrahedron_info * tetra_info_0, //Info of each original tetrahedron (aabb, barycentric matrix, etc)
                                     const unsigned * indexes,              //Stores, for each cuda block, its corresponding aabb_info index.
                                     VOXELTYPE * cube_1,                    //Deformed voxel cube [OUT].
                                     const int3 cube_dims_1,                //Dimensions of the deformed cube (width, height, depth).
                                     const tetrahedron_info * tetra_info_1, //Info of each deformed tetrahedron (aabb, barycentric matrix, etc)
                                     const aabb_info * aabb_info_1,         //Info of each aabb (aabb, dims, inferior corner, etc)
                                     const unsigned batch_start_block_offset); //Start block offset of this batch execution.


void cuda_rasterize_tex3DCall(const unsigned num_blocks,
                              const tetrahedron_info * tetra_info_0,
                              const unsigned * indexes,
                              VOXELTYPE * cube_1,
                              //cudaArray * cuda_image_array,
                              const int3 cube_dims_1,
                              const tetrahedron_info * tetra_info_1,
                              const aabb_info * aabb_info_1);

/*

__global__ void cuda_check_barycentrics_kernel(const tetrahedron_info * tetra_info_0, //Info of each original tetrahedron (aabb, barycentric matrix, etc)
                                               const unsigned * indexes,              //Stores, for each cuda block, its corresponding aabb_info index.
                                               voxel_cube::VOXELTYPE * cube_1,  //Deformed voxel cube [OUT].
                                               const int3 cube_dims_1,                //Dimensions of the deformed cube (width, height, depth).
                                               const tetrahedron_info * tetra_info_1, //Info of each deformed tetrahedron (aabb, barycentric matrix, etc)
                                               const aabb_info * aabb_info_1,         //Info of each aabb (aabb, dims, inferior corner, etc)
                                               const unsigned batch_start_block_offset); //Start block offset of this batch execution.


void cuda_check_barycentrics_call(const unsigned num_blocks,
                                  const tetrahedron_info * tetra_info_0,
                                  const unsigned * indexes,
                                  voxel_cube::VOXELTYPE * cube_1,
                                  const int3 cube_dims_1,
                                  const tetrahedron_info * tetra_info_1,
                                  const aabb_info * aabb_info_1);


void cuda_measure_useful_threads_call(const unsigned num_blocks,
                                      const tetrahedron_info * tetra_info_0,
                                      const unsigned * indexes,
                                      unsigned char *useful_threads_flags,
                                      const int3 cube_dims_1,
                                      const tetrahedron_info *tetra_info_1,
                                      const aabb_info *aabb_info_1);




__global__ void cuda_measure_useful_threads_kernel(const unsigned *indexes, //Info of each original tetrahedron (aabb, barycentric matrix, etc)
                                                   unsigned char *useful_threads_flags,              //Stores, for each cuda block, its corresponding aabb_info index.
                                                   const tetrahedron_info *tetra_info_1,
                                                   const aabb_info *aabb_info_1,                //Dimensions of the deformed cube (width, height, depth).
                                                   const unsigned batch_start_block_offset); //Info of each deformed tetrahedron (aabb, barycentric matrix, etc)

*/

#endif // RASTERIZE_H



