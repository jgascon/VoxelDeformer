#include <stdio.h>
#include <voxelVolume.hpp>
#include <types.hpp>
#include <matrix.hpp>
#include <culler_rasterizer.cuh>


#if __CUDA_ARCH__ < 200
    #define printf   //
#endif

cudaArray * d_input_volumeArray = 0;
texture<VOXELTYPE, 3, cudaReadModeNormalizedFloat> input_tex;

cudaArray * d_output_volumeArray = 0;
surface<void, cudaSurfaceType3D> output_tex;





void loadCudaInputTexture3D(const VOXELTYPE * h_volume,
                                 const unsigned width,
                                 const unsigned height,
                                 const unsigned depth) {
    cudaExtent volumeSize;
    volumeSize.width = width;
    volumeSize.height = height;
    volumeSize.depth = depth;

    cudaError cuda_error;
    // create 3D array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VOXELTYPE>();
    //cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0,
    //                                                          cudaChannelFormatKindUnsigned);

    cuda_error = cudaMalloc3DArray(&d_input_volumeArray, &channelDesc, volumeSize);
    if (cuda_error != cudaSuccess) {
        const char * error_string = cudaGetErrorString(cuda_error);
        cerr << "ERROR CUDA: loadCudaTexture3D --> error code: ["<< cuda_error << "] text: [" << error_string <<"]." << endl;
        exit(-1);
    }

    // copy data to 3D array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr((void *)h_volume,
                                            volumeSize.width * sizeof(VOXELTYPE),
                                            volumeSize.width,
                                            volumeSize.height);
    copyParams.dstArray = d_input_volumeArray;
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyHostToDevice;
    cuda_error = cudaMemcpy3D(&copyParams);
    if (cuda_error != cudaSuccess) {
        const char * error_string = cudaGetErrorString(cuda_error);
        cerr << "ERROR CUDA: loadCudaTexture3D --> error code: ["<< cuda_error << "] text: [" << error_string <<"]." << endl;
        exit(-1);
    }

    // set texture parameters
    input_tex.normalized = false;                      // access with normalized texture coordinates
    //tex.filterMode = cudaFilterModePoint;              // nearest interpolation.
    input_tex.filterMode = cudaFilterModeLinear;       // linear interpolation
    input_tex.addressMode[0] = cudaAddressModeClamp;   // wrap texture coordinates
    input_tex.addressMode[1] = cudaAddressModeClamp;
    input_tex.addressMode[2] = cudaAddressModeClamp;

    // bind array to 3D texture
    cuda_error = cudaBindTextureToArray(input_tex, d_input_volumeArray, channelDesc);
    if (cuda_error != cudaSuccess) {
        const char * error_string = cudaGetErrorString(cuda_error);
        cerr << "ERROR CUDA: loadCudaTexture3D --> error code: ["<< cuda_error << "] text: [" << error_string <<"]." << endl;
        exit(-1);
    }
} //void loadCudaInputTexture3D



void unLoadCudaInputTexture3D() {
    cudaUnbindTexture(input_tex);
    cudaFreeArray(d_input_volumeArray);
} //void unLoadCudaInputTexture3D()






void createCudaOutputTexture3D(const unsigned width,
                               const unsigned height,
                               const unsigned depth) {
    cudaExtent volumeSize;
    volumeSize.width = width;
    volumeSize.height = height;
    volumeSize.depth = depth;

    // Allocate array and copy image data
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VOXELTYPE>();

    cudaMalloc3DArray(&d_output_volumeArray,
                      &channelDesc,
                      volumeSize,
                      cudaArraySurfaceLoadStore);

    cudaBindSurfaceToArray(output_tex, d_output_volumeArray);

} //void createCudaOutputTexture3D




void destroyCudaOutputTexture3D() {
    //cudaFree(dData);
    cudaFreeArray(d_output_volumeArray);
} //void destroyCudaOutputTexture3D()




__device__ float3 mul(const float * M, const float3 p) {
    float3 r;
    r.x = M[0] * p.x + M[1] * p.y + M[2] * p.z;
    r.y = M[3] * p.x + M[4] * p.y + M[5] * p.z;
    r.z = M[6] * p.x + M[7] * p.y + M[8] * p.z;
    return r;
}





//Given a 3D point inside a voxel cube and a color value, assigns that color value to the corresponding voxel.
__device__ void Set(const float3 p,
                    const int3 cube_dims,
                    VOXELTYPE * cube,
                    const VOXELTYPE color) {
    //Assuming that each coordinate is inside the cube.
    if (p.x >= 0 && p.x < cube_dims.x &&
        p.y >= 0 && p.y < cube_dims.y &&
        p.z >= 0 && p.z < cube_dims.z) {
        int3 a;
        a.x = p.x;
        a.y = p.y;
        a.z = p.z;
        int offset = cube_dims.x * ((cube_dims.y * a.z) + a.y) + a.x;
        cube[offset] = color;
    }
}

__device__ float3 coords_to_barycentrics(const tetrahedron_info my_tetra_info,
                                         const float3 P) {
    float3 Q;
    Q.x = P.x - my_tetra_info.node_0.x;
    Q.y = P.y - my_tetra_info.node_0.y;
    Q.z = P.z - my_tetra_info.node_0.z;

    return mul(my_tetra_info.bar_mat, Q);
}


__device__ float3 barycentrics_to_coords(const tetrahedron_info my_tetra_info,
                                         const float3 bar) {
    float3 result = mul(my_tetra_info.bar_mat, bar);
    result.x += my_tetra_info.node_0.x;
    result.y += my_tetra_info.node_0.y;
    result.z += my_tetra_info.node_0.z;
    return result;
}





void create_ibox_labeled_grid(iaabb bb,
                              uvec3 new_aabbs_dims,
                              vector<culling_grid_node> & result_nodes,
                              unsigned start_grid_nodes_index,
                              vector<uvec8> & result_boxes,
                              unsigned start_grid_boxes_index,
                              unsigned node_label) {

    uvec3 steps(CEIL_POS(float(bb.width()) / float(new_aabbs_dims.x)),
                CEIL_POS(float(bb.height()) / float(new_aabbs_dims.y)),
                CEIL_POS(float(bb.depth()) / float(new_aabbs_dims.z)) );

    unsigned node_id = start_grid_nodes_index;

    culling_grid_node p;
    p.tetra_i = node_label;

    for (unsigned k = 0; k <= steps.z; k++) {
        p.node.z = bb.aabb_min.z + new_aabbs_dims.z * k;

        for (unsigned j=0; j <= steps.y; j++) {
            p.node.y = bb.aabb_min.y + new_aabbs_dims.y * j;

            for (unsigned i=0; i <= steps.x; i++) {
                p.node.x = bb.aabb_min.x + new_aabbs_dims.x * i;
                result_nodes[node_id++] = p;
            }
        }
    }

    //Creating the "grid_boxes"
    uvec8 box;
    unsigned box_id = start_grid_boxes_index;

    for (unsigned k = 0; k < steps.z; k++) {
        for (unsigned j=0; j < steps.y; j++) {
            for (unsigned i=0; i < steps.x; i++) {

                box[0] = (k * (steps.y + 1) + j) * (steps.x + 1) + i + start_grid_nodes_index;
                box[1] = box[0] + 1;

                box[2] = box[0] + (steps.x + 1);
                box[3] = box[2] + 1;

                box[4] = box[0] + (steps.x + 1) * (steps.y + 1);
                box[5] = box[4] + 1;

                box[6] = box[4] + (steps.x + 1);
                box[7] = box[6] + 1;

                result_boxes[box_id++] = box;
            }
        }
    }
}



/*

__device__ unsigned char cuda_aux_set_barycentrics_mask(const float3 & bar) {
    //mask is an 8 bits number where each bit represents an outside indicator.
    // bit 0 --> bar.x < 0.0
    // bit 1 --> bar.x > 1.0

    // bit 2 --> bar.y < 0.0
    // bit 3 --> bar.y > 1.0

    // bit 4 --> bar.z < 0.0
    // bit 5 --> bar.z > 1.0

    // bit 6 --> (1.0 - bar.x - bar.y - bar.z) < 0.0
    // bit 7 --> (1.0 - bar.x - bar.y - bar.z) > 1.0

    //If mask == 0x00 --> The aabb is intersecting with the tetrahedron.
    //If mask != 0x00 --> The aabb is completely outside of the tetrahedron.
    unsigned char result = 0x00;
    if (bar.x < -0.00001) {
        result |= 0x80;
    }
    if (bar.x > 1.00001) {
        result |= 0x40;
    }

    if (bar.y < -0.00001) {
        result |= 0x20;
    }
    if (bar.y > 1.00001) {
        result |= 0x10;
    }

    if (bar.z < -0.00001) {
        result |= 0x08;
    }
    if (bar.z > 1.00001) {
        result |= 0x04;
    }

    double w = 1.0 - bar.x - bar.y - bar.z;
    if (w < -0.00001) {
        result |= 0x02;
    }
    if (w > 1.00001) {
        result |= 0x01;
    }

    return result;
}


__global__ void cuda_compute_nodes_masks(const unsigned num_grid_nodes,
                                         const tetrahedron_info * tetra_info_1,
                                         const culling_grid_node * grid_nodes,
                                         unsigned char * result_nodes_masks) {

    int i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    if (i >= num_grid_nodes) {
        return;
    }

    culling_grid_node n = grid_nodes[i];
    tetrahedron_info t = tetra_info_1[n.tetra_i];

    float3 p, bar;
    p.x = n.node.x - t.node_0.x;
    p.y = n.node.y - t.node_0.y;
    p.z = n.node.z - t.node_0.z;

    bar = mul(t.bar_mat, p);

    unsigned char mask = cuda_aux_set_barycentrics_mask(bar);

//    if (i == 94) {
//        printf("\n\nCUDA node %d coords (%d, %d, %d) tetra %d bar (%f, %f, %f) mask %d sum %f" << endl;,
//                i, n.node.x, n.node.y, n.node.z, n.tetra_i, bar.x, bar.y, bar.z, mask, bar.x+bar.y+bar.z);
//        printf("Node 0: (%f, %f, %f)" << endl;, t.node_0.x, t.node_0.y, t.node_0.z);
//        printf("Deformed matrix:" << endl;);
//        for (unsigned mat_i = 0; mat_i < 3; mat_i++) {
//            printf("%f %f %f" << endl;, t.bar_mat[mat_i * 3], t.bar_mat[mat_i*3+1], t.bar_mat[mat_i*3+2]);
//        }
//        printf("" << endl;);
//    }

    result_nodes_masks[i] = mask;

} //cuda_compute_nodes_masks




void cuda_compute_nodes_masks_call(const unsigned num_grid_nodes,
                                   const tetrahedron_info * tetra_info_1,
                                   const culling_grid_node * grid_nodes,
                                   unsigned char * result_nodes_masks) {

    unsigned remainding_blocks = CEIL(float(num_grid_nodes)/float(THREADS_PER_BLOCK));
    unsigned batch_num_blocks = 0;

    while (remainding_blocks > 0) {

        if (remainding_blocks < 65535) {
            batch_num_blocks = remainding_blocks;
        } else {
            batch_num_blocks = 65535;
        }

        //cout << "cuda_compute_nodes_masks_call: Running batch of size " << batch_num_blocks << endl;
        cuda_compute_nodes_masks <<< batch_num_blocks, THREADS_PER_BLOCK >>> (num_grid_nodes,
                                                                              tetra_info_1,
                                                                              grid_nodes,
                                                                              result_nodes_masks);
        cudaError_t cuda_result = cudaGetLastError();
        if (cuda_result != cudaSuccess) {
            const char * error_string = cudaGetErrorString(cuda_result);
            cerr << "ERROR CUDA: cuda_compute_nodes_masks_call: Kernel failure code ["<< cuda_result << "] = [" << error_string <<"]."<<endl;
            return;
        }
        remainding_blocks -= batch_num_blocks;
    }
} //cuda_compute_nodes_masks_call





__global__ void cuda_compute_boxes_masks(const unsigned num_grid_boxes,
                                         const unsigned char * grid_nodes_masks,
                                         unsigned * grid_boxes,
                                         unsigned char * result_boxes_masks) {

    int i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    if (i >= num_grid_boxes) {
        return;
    }

    unsigned * box = &grid_boxes[i*8];
//    unsigned char box_mask = grid_nodes_masks[box[0]];
//    for (unsigned j=1; j<8 && box_mask != 0x00; j++) {
//        box_mask &= grid_nodes_masks[box[j]];
//    }
//True means "culled".
//    result_boxes_masks[i] = box_mask;

    result_boxes_masks[i] = grid_nodes_masks[box[0]] & grid_nodes_masks[box[1]] &
                            grid_nodes_masks[box[2]] & grid_nodes_masks[box[3]] &
                            grid_nodes_masks[box[4]] & grid_nodes_masks[box[5]] &
                            grid_nodes_masks[box[6]] & grid_nodes_masks[box[7]];

} //cuda_compute_boxes_masks





void cuda_compute_boxes_masks_call(const unsigned num_grid_boxes,
                                   const unsigned char * grid_nodes_masks,
                                   unsigned *grid_boxes,
                                   unsigned char * result_boxes_masks) {

    unsigned remainding_blocks = CEIL(float(num_grid_boxes)/float(THREADS_PER_BLOCK));
    unsigned batch_num_blocks = 0;

    while (remainding_blocks > 0) {
        if (remainding_blocks < 65535) {
            batch_num_blocks = remainding_blocks;
        } else {
            batch_num_blocks = 65535;
        }

        //cout << "cuda_compute_boxes_masks_call: Running batch of size " << batch_num_blocks << endl;
        cuda_compute_boxes_masks <<< batch_num_blocks, THREADS_PER_BLOCK >>> (num_grid_boxes,
                                                                              grid_nodes_masks,
                                                                              grid_boxes,
                                                                              result_boxes_masks);
        cudaError_t cuda_result = cudaGetLastError();
        if (cuda_result != cudaSuccess) {
            const char * error_string = cudaGetErrorString(cuda_result);
            cerr << "ERROR CUDA: cuda_compute_boxes_masks_call: Kernel failure code ["<< cuda_result << "] = [" << error_string <<"]."<<endl;
            return;
        }
        remainding_blocks -= batch_num_blocks;
    }
} //cuda_compute_boxes_masks_call
*/




void cuda_upload_tetrahedra_info(const vector<tetrahedron_info> & tetra_info_1,
                                 tetrahedron_info ** d_tetra_info_1) {
    //Upload tetrahedra_info_1 to CUDA:
    cudaError cuda_result = cudaMalloc((void**) d_tetra_info_1,
                                       tetra_info_1.size() * sizeof(tetrahedron_info));
    if (cuda_result != cudaSuccess) {
        cerr << "ERROR CUDA: cuda_upload_tetrahedra_info cuda malloc failed" << endl;
        return; //exit(-1);
    }
    cuda_result = cudaMemcpy(*d_tetra_info_1,
                             (void *) tetra_info_1.data(),
                             tetra_info_1.size() * sizeof(tetrahedron_info),
                             cudaMemcpyHostToDevice);
    if (cuda_result != cudaSuccess) {
        cerr << "ERROR CUDA: cuda_upload_tetrahedra_info cuda memcopy failed" << endl;
        return; //exit(-1);
    }
}



//Clear tetrahedra info to GPU.
void cuda_clear_tetrahedra_info(tetrahedron_info * d_tetra_info_1) {
    if (d_tetra_info_1 != NULL) {
        cudaError  cuda_result = cudaFree(d_tetra_info_1);
        if (cuda_result != cudaSuccess) {
            cerr << "ERROR CUDA: cuda_clear_tetrahedra_info --> d_tetra_info_1 cuda free failed" << endl;
            return;
        }
        d_tetra_info_1 = NULL;
    }
}




__global__ void cuda_rasterize3D(const tetrahedron_info * tetra_info_0, //Info of each original tetrahedron (aabb, barycentric matrix, etc)
                                 const unsigned * indexes,              //Stores, for each cuda block, its corresponding aabb_info index.
                                 VOXELTYPE * cube_1,  //Deformed voxel cube [OUT].
                                 const int3 cube_dims_1,                //Dimensions of the deformed cube (width, height, depth).
                                 const tetrahedron_info * tetra_info_1, //Info of each deformed tetrahedron (aabb, barycentric matrix, etc)
                                 const aabb_info * aabb_info_1,         //Info of each aabb (aabb, dims, inferior corner, etc)
                                 const unsigned batch_start_block_offset) { //Start block offset of this batch execution.

    int cell_id = indexes[blockIdx.x + batch_start_block_offset];

    //Getting the aabb where I am and the tetrahedron where it belongs.
    aabb_info cell = aabb_info_1[cell_id];

    //Getting the 3D coords inside the aabb.
    //unsigned global_block_offset = blockIdx.x + batch_start_block_offset;
    //unsigned aabb_local_block_offset = global_block_offset - cell.start_block_offset;
    //unsigned aabb_local_voxel_offset = aabb_local_block_offset * THREADS_PER_BLOCK + threadIdx.x;

    //Discarding voxels outside aabb.
    //if (aabb_local_voxel_offset >= cell.dims.x * cell.dims.y * cell.dims.z) {
    //if (aabb_local_voxel_offset >= aabbs_dims.x * aabbs_dims.y * aabbs_dims.z) {
    //    return;
    //}

    tetrahedron_info my_tetra_info_1 = tetra_info_1[cell.index];

    //float3 P_1 = offset_to_coords(aabb_local_voxel_offset, cell, aabbs_dims);

    //Computing the 3d position of the voxel.
    float3 P_1;
    P_1.x = cell.iaabb_min.x + threadIdx.x;
    P_1.y = cell.iaabb_min.y + threadIdx.y;
    P_1.z = cell.iaabb_min.z + threadIdx.z;

    //Getting the Barycentric coordinates inside the tetrahedron.
    float3 bar_1 = coords_to_barycentrics(my_tetra_info_1, P_1);

    //Is P inside the current tetrahedron? let's check the barycentric coordinates.
    if ((bar_1.x + bar_1.y + bar_1.z) <= 1.0 && bar_1.x >= 0.0 && bar_1.y >= 0.0 && bar_1.z >= 0.0) {
        //Getting the information of the undeformed tetrahedron.
        tetrahedron_info my_tetra_info_0 = tetra_info_0[cell.index];

        //Computing the 3D coords inside the original voxel cube.
        float3 P_0 = barycentrics_to_coords(my_tetra_info_0, bar_1);

        //Getting the voxel color from cube_0
        //VOXELTYPE color = tex3D(tex, P_0.x, P_0.y, P_0.z) * 32768;
        VOXELTYPE color = tex3D(input_tex, P_0.x, P_0.y, P_0.z) * 256;

        //Setting the color to cube_1.
        Set(P_1, cube_dims_1, cube_1, color);
    }
} //void cuda_rasterize3D




void cuda_rasterize3DCall(const unsigned num_blocks,
                          const tetrahedron_info * tetra_info_0,
                          const unsigned * indexes,
                          VOXELTYPE * cube_1,
                          const int3 cube_dims_1,
                          const tetrahedron_info * tetra_info_1,
                          const aabb_info * aabb_info_1) {

    unsigned remainding_blocks = num_blocks;
    unsigned start_block_offset = 0;
    unsigned batch_num_blocks = 0;

    dim3 threadsBlock(8, 8, 8);

    while (remainding_blocks > 0) {
        if (remainding_blocks < 65535) {
            batch_num_blocks = remainding_blocks;
        } else {
            batch_num_blocks = 65535;
        }

        //cout << "Rasterize: Running batch of size " << batch_num_blocks << endl;

        //cuda_rasterize3D <<< batch_num_blocks, THREADS_PER_BLOCK >>> (tetra_info_0,
        cuda_rasterize3D <<< batch_num_blocks, threadsBlock >>> (tetra_info_0,
                                                                 indexes,
                                                                 cube_1,
                                                                 cube_dims_1,
                                                                 tetra_info_1,
                                                                 aabb_info_1,
                                                                 start_block_offset);
        cudaError cuda_result = cudaGetLastError();
        if (cuda_result != cudaSuccess) {
            const char * error_string = cudaGetErrorString(cuda_result);
            cerr << "ERROR CUDA: NEW: Kernel failure with code ["<< cuda_result << "] = [" << error_string <<"]."<<endl;
            return;
        }

        start_block_offset += batch_num_blocks;
        remainding_blocks -= batch_num_blocks;
    }
} //void cuda_rasterize3DCall






__global__ void cuda_rasterize_tex3D(const tetrahedron_info * tetra_info_0, //Info of each original tetrahedron (aabb, barycentric matrix, etc)
                                     const unsigned * indexes,              //Stores, for each cuda block, its corresponding aabb_info index.
                                     VOXELTYPE * cube_1,  //Deformed voxel cube [OUT].
                                     const int3 cube_dims_1,                //Dimensions of the deformed cube (width, height, depth).
                                     const tetrahedron_info * tetra_info_1, //Info of each deformed tetrahedron (aabb, barycentric matrix, etc)
                                     const aabb_info * aabb_info_1,         //Info of each aabb (aabb, dims, inferior corner, etc)
                                     const unsigned batch_start_block_offset) { //Start block offset of this batch execution.

    int cell_id = indexes[blockIdx.x + batch_start_block_offset];

    //Getting the aabb where I am and the tetrahedron where it belongs.
    aabb_info cell = aabb_info_1[cell_id];

    //Getting the 3D coords inside the aabb.
    //unsigned global_block_offset = blockIdx.x + batch_start_block_offset;
    //unsigned aabb_local_block_offset = global_block_offset - cell.start_block_offset;
    //unsigned aabb_local_voxel_offset = aabb_local_block_offset * THREADS_PER_BLOCK + threadIdx.x;

    //Discarding voxels outside aabb.
    //if (aabb_local_voxel_offset >= cell.dims.x * cell.dims.y * cell.dims.z) {
    //if (aabb_local_voxel_offset >= aabbs_dims.x * aabbs_dims.y * aabbs_dims.z) {
    //    return;
    //}

    tetrahedron_info my_tetra_info_1 = tetra_info_1[cell.index];

    //float3 P_1 = offset_to_coords(aabb_local_voxel_offset, cell, aabbs_dims);

    //Computing the 3d position of the voxel.
    float3 P_1;
    P_1.x = cell.iaabb_min.x + threadIdx.x;
    P_1.y = cell.iaabb_min.y + threadIdx.y;
    P_1.z = cell.iaabb_min.z + threadIdx.z;

    //Getting the Barycentric coordinates inside the tetrahedron.
    float3 bar_1 = coords_to_barycentrics(my_tetra_info_1, P_1);

    //Is P inside the current tetrahedron? let's check the barycentric coordinates.
    if ((bar_1.x + bar_1.y + bar_1.z) <= 1.0 && bar_1.x >= 0.0 && bar_1.y >= 0.0 && bar_1.z >= 0.0) {
        //Getting the information of the undeformed tetrahedron.
        tetrahedron_info my_tetra_info_0 = tetra_info_0[cell.index];

        //Computing the 3D coords inside the original voxel cube.
        float3 P_0 = barycentrics_to_coords(my_tetra_info_0, bar_1);

        //Getting the voxel color from cube_0
        //VOXELTYPE color = tex3D(tex, P_0.x, P_0.y, P_0.z) * 32768;
        VOXELTYPE color = tex3D(input_tex, P_0.x, P_0.y, P_0.z) * 256;

        //Setting the color to cube_1.
        Set(P_1, cube_dims_1, cube_1, color);

        surf3Dwrite(color, output_tex, P_1.x * sizeof(VOXELTYPE), P_1.y, P_1.z, cudaBoundaryModeTrap);
    }
} //void cuda_rasterize_tex3D




void cuda_rasterize_tex3DCall(const unsigned num_blocks,
                              const tetrahedron_info * tetra_info_0,
                              const unsigned * indexes,
                              VOXELTYPE * cube_1,
                              const int3 cube_dims_1,
                              const tetrahedron_info * tetra_info_1,
                              const aabb_info * aabb_info_1) {

    unsigned remainding_blocks = num_blocks;
    unsigned start_block_offset = 0;
    unsigned batch_num_blocks = 0;

    dim3 threadsBlock(8, 8, 8);

    while (remainding_blocks > 0) {
        if (remainding_blocks < 65535) {
            batch_num_blocks = remainding_blocks;
        } else {
            batch_num_blocks = 65535;
        }

        //cout << "Rasterize: Running batch of size " << batch_num_blocks << endl;

        //cuda_rasterize3D <<< batch_num_blocks, THREADS_PER_BLOCK >>> (tetra_info_0,
        cuda_rasterize3D <<< batch_num_blocks, threadsBlock >>> (tetra_info_0,
                                                                 indexes,
                                                                 cube_1,
                                                                 cube_dims_1,
                                                                 tetra_info_1,
                                                                 aabb_info_1,
                                                                 start_block_offset);
        cudaError cuda_result = cudaGetLastError();
        if (cuda_result != cudaSuccess) {
            const char * error_string = cudaGetErrorString(cuda_result);
            cerr << "ERROR CUDA: NEW: Kernel failure with code ["<< cuda_result << "] = [" << error_string <<"]."<<endl;
            return;
        }

        start_block_offset += batch_num_blocks;
        remainding_blocks -= batch_num_blocks;
    }

    cudaDeviceSynchronize();

} //void cuda_rasterize_tex3DCall






void update_tetrahedra_info_from_tetramesh(vector<tetrahedron_info> & result,
                                           const vector<uvec4> & tetras,
                                           const vector<vec3> & nodes,
                                           bool invert_matrices) {
    uvec4 t;
    fmatrix mat;
    aabb bb;
    tetrahedron_info cell;
    result.clear();
    result.resize(tetras.size());
    vec3 tetra_nodes[4];

    for (unsigned tetra_i=0; tetra_i<tetras.size(); tetra_i++) {
        t = tetras[tetra_i];
        //Collect all the nodes of this tetra.
        for (unsigned node_i=0; node_i<4; node_i++) {
            tetra_nodes[node_i] = nodes[t[node_i]];
        }

//FIXME: Dilating the tetra nodes is good for adding frontier voxels, but very bad for culling.
        //  dilate(tetra_nodes, 1.01);

        //Making the AABB of deformed tetrahedron.
        cell.node_0.x = tetra_nodes[0].x;
        cell.node_0.y = tetra_nodes[0].y;
        cell.node_0.z = tetra_nodes[0].z;

        bb.set(tetra_nodes[0]);
        for (unsigned node_i=1; node_i<4; node_i++) {
            bb.add(tetra_nodes[node_i]);
        }

        //New integer arithmetic.
        cell.iaabb_min.x = FLOOR(bb.aabb_min.x);
        cell.iaabb_min.y = FLOOR(bb.aabb_min.y);
        cell.iaabb_min.z = FLOOR(bb.aabb_min.z);
        cell.dims.x = CEIL(bb.aabb_max.x) - cell.iaabb_min.x;
        cell.dims.y = CEIL(bb.aabb_max.y) - cell.iaabb_min.y;
        cell.dims.z = CEIL(bb.aabb_max.z) - cell.iaabb_min.z;

        //Making the tetrahedron deformed matrix.
        mat.map(3, 3, cell.bar_mat);
        for (unsigned node_i=1; node_i<4; node_i++) {
            for (unsigned k=0; k<3; k++) {
                mat[k][node_i-1] = tetra_nodes[node_i][k] - tetra_nodes[0][k];
            }
        }
        if (invert_matrices == true) {
            mat.invert();
        }
        mat.unmap();
        result[tetra_i] = cell;
    }
}





vec3 aabb_info::get_ith_corner(const unsigned i, const uvec3 aabb_dims) const {
    vec3 corner;
    switch(i) {
        case 0:
            corner.x = this->iaabb_min.x;
            corner.y = this->iaabb_min.y;
            corner.z = this->iaabb_min.z;
            break;

        case 1:
            corner.x = this->iaabb_min.x;
            corner.y = this->iaabb_min.y;
            corner.z = this->iaabb_min.z + aabb_dims.z;
            break;

        case 2:
            corner.x = this->iaabb_min.x;
            corner.y = this->iaabb_min.y + aabb_dims.y;
            corner.z = this->iaabb_min.z;
            break;

        case 3:
            corner.x = this->iaabb_min.x;
            corner.y = this->iaabb_min.y + aabb_dims.y;
            corner.z = this->iaabb_min.z + aabb_dims.z;
            break;

        case 4:
            corner.x = this->iaabb_min.x + aabb_dims.x;
            corner.y = this->iaabb_min.y;
            corner.z = this->iaabb_min.z;
            break;

        case 5:
            corner.x = this->iaabb_min.x + aabb_dims.x;
            corner.y = this->iaabb_min.y;
            corner.z = this->iaabb_min.z + aabb_dims.z;
            break;

        case 6:
            corner.x = this->iaabb_min.x + aabb_dims.x;
            corner.y = this->iaabb_min.y + aabb_dims.y;
            corner.z = this->iaabb_min.z;
            break;

        default:
            corner.x = this->iaabb_min.x + aabb_dims.x;
            corner.y = this->iaabb_min.y + aabb_dims.y;
            corner.z = this->iaabb_min.z + aabb_dims.z;
            break;
    }
    return corner;
}






void update_aabb_info_from_tetra_info(vector<aabb_info> & result,
                                      const vector<tetrahedron_info> & tetras,
                                      const uvec3 aabb_dims,
                                      const unsigned threads_per_block) {

    //There are as many aabbs as tetrahedra are.
    result.resize(tetras.size());
    unsigned current_offset = 0;

    for (unsigned tetra_i=0; tetra_i < tetras.size(); tetra_i++) {
        const tetrahedron_info & tetra = tetras[tetra_i];
        aabb_info & bb = result[tetra_i];
        bb.iaabb_min = tetra.iaabb_min;
        //bb.dims = tetra.aabb_dims;
        bb.index = tetra_i;
        bb.start_block_offset = current_offset;
        current_offset += unsigned(CEIL_POS(float(aabb_dims.x * aabb_dims.y * aabb_dims.z) / float(threads_per_block)));
    }
}




