#include <voxelDeformer.hpp>


/*
///Original voxel volume for this deformer.
voxelVolume * vol_0;

///Original voxel volume aabb for this deformer.
iaabb volume_aabb_0;

///Original tetramesh for this deformer.
tetramesh * m_0;

///Original tetramesh aabb for this deformer.
aabb tetramesh_aabb_0;
*/

voxelDeformer::voxelDeformer() {
    this->finish();
}


voxelDeformer::~voxelDeformer() {
    this->finish();
}

/*
aabb voxelDeformer::world_to_voxel_space(const aabb & bb) const {
    aabb r;
    r.aabb_min = world_to_device_scale.scale(bb.aabb_min);
    r.aabb_max = world_to_device_scale.scale(bb.aabb_max);
    return r;
} //iaabb voxel_viewer::world_to_voxel_space(const aabb & bb);
*/



void voxelDeformer::init(voxelVolume * volume, tetramesh * tetra) {
    if (!volume || !tetra) {
        cerr << "ERROR: voxelDeformer::init --> one or more params are NULL!" << endl;
        return;
    }
    if (this->vol_0 != NULL) {
        this->finish();
    }
    this->vol_0 = volume;
    this->volume_aabb_0.set(ivec3(0,0,0));

    this->volume_aabb_0.add(ivec3(volume->width(), volume->height(), volume->depth()));

    this->m_0 = tetra;
    this->tetramesh_aabb_0.set(tetra->x);


    //-------------- Init Cuda Culler Rasterizer --------------

    //Prefixes:
    //         h_ : "host" variables, which live in the CPU.
    //         d_ : "device" variables, which live in the CPU.
    cudaError cuda_result;

    //Init CUDA GPU:

    //TODO: Study how to select the most powerful GPU, if many.
    cudaSetDevice(0);
    //cudaGLSetGLDevice(0);

    //Upload cube_0 to GPU.
    loadCudaInputTexture3D(this->vol_0->texels().data(),
                           this->vol_0->width(),
                           this->vol_0->height(),
                           this->vol_0->depth());

    //Compute "tetrahedra_info" from initial mesh

    //Convert all the nodes positions to voxel space.
    vector<vec3> nodes;
    nodes.resize(this->m_0->x.size());

    vec3 ratio(this->volume_aabb_0.width()  / this->tetramesh_aabb_0.width(),
               this->volume_aabb_0.height() / this->tetramesh_aabb_0.height(),
               this->volume_aabb_0.depth()  / this->tetramesh_aabb_0.depth());

    for (unsigned i=0; i<nodes.size(); i++) {
        vec3 & n0 = this->m_0->x[i];
        vec3 & n = nodes[i];

        n.x = (n0.x - this->tetramesh_aabb_0.aabb_min.x) * ratio.x;
        n.y = (n0.y - this->tetramesh_aabb_0.aabb_min.y) * ratio.y;
        n.z = (n0.z - this->tetramesh_aabb_0.aabb_min.z) * ratio.z;
    }

    update_tetrahedra_info_from_tetramesh(this->h_tetrahedra_info_0,
                                          this->m_0->t,
                                          nodes);

    //Upload tetrahedra_info to CUDA:
    cuda_result = cudaMalloc((void**) &d_tetrahedra_info_0,
                             h_tetrahedra_info_0.size() * sizeof(tetrahedron_info));
    if (cuda_result != cudaSuccess) {
        cerr << "ERROR CUDA: h_tetrahedra_info_0 cuda malloc failed\n";
        return; //exit(-1);
    }
    cuda_result = cudaMemcpyAsync(d_tetrahedra_info_0,
                                  (void *) h_tetrahedra_info_0.data(),
                                  h_tetrahedra_info_0.size() * sizeof(tetrahedron_info),
                                  cudaMemcpyHostToDevice);
    if (cuda_result != cudaSuccess) {
        cerr << "ERROR CUDA: h_tetrahedra_info_0 cuda memcopy failed\n";
        return; //exit(-1);
    }
}



void voxelDeformer::deform(const vector<vec3> & nodes_1, voxelVolume & volume_out) {
    if (this->vol_0 == NULL || this->vol_0 == NULL) {
        cerr << "ERROR: voxelDeformer::deform --> voxelDeformer not initialized, call init before!" << endl;
        return;
    }
    //Computing the size of the new voxel cube.
    this->tetramesh_aabb_1.set(nodes_1);
    vec3 ratio(tetramesh_aabb_1.width() / this->tetramesh_aabb_0.width(),
               tetramesh_aabb_1.height() / this->tetramesh_aabb_0.height(),
               tetramesh_aabb_1.depth() / this->tetramesh_aabb_0.depth());


//    cout << "old tetramesh aabb " << this->tetramesh_aabb_0.to_s() << endl;
//    cout << "new tetramesh aabb " << tetramesh_aabb_1.to_s() << endl;
//    cout << "ratio " << ratio.to_s() << endl;

    this->volume_aabb_1.aabb_min.set(0,0,0);
    this->volume_aabb_1.aabb_max.set(int(ceil(this->volume_aabb_0.aabb_max.x * ratio.x)),
                               int(ceil(this->volume_aabb_0.aabb_max.y * ratio.y)),
                               int(ceil(this->volume_aabb_0.aabb_max.z * ratio.z)));

    //cout << "old volume aabb " << this->volume_aabb_0.to_s() << endl;
    //cout << "new volume aabb " << volume_aabb_1.to_s() << endl << endl;

    volume_out.resize(this->volume_aabb_1.aabb_max.x,
                      this->volume_aabb_1.aabb_max.y,
                      this->volume_aabb_1.aabb_max.z);


    createCudaOutputTexture3D(this->volume_aabb_1.aabb_max.x,
                              this->volume_aabb_1.aabb_max.y,
                              this->volume_aabb_1.aabb_max.z);


    //------------------ Launch cuda culler Rasterizer ------------------

    //Typical cuda block of 512, which is 8x8x8.
    uvec3 mini_aabbs_dims = uvec3(8, 8, 8);


    //---------------- Convert all the nodes positions to voxel space ----------------

    //Convert all the nodes positions to voxel space.
    vector<vec3> nodes;
    nodes.resize(nodes_1.size());

    ratio.set(this->volume_aabb_1.width()  / tetramesh_aabb_1.width(),
              this->volume_aabb_1.height() / tetramesh_aabb_1.height(),
              this->volume_aabb_1.depth()  / tetramesh_aabb_1.depth());

    for (unsigned i=0; i<nodes.size(); i++) {
        const vec3 & n0 = nodes_1[i];
        vec3 & n = nodes[i];

        n.x = (n0.x - tetramesh_aabb_1.aabb_min.x) * ratio.x;
        n.y = (n0.y - tetramesh_aabb_1.aabb_min.y) * ratio.y;
        n.z = (n0.z - tetramesh_aabb_1.aabb_min.z) * ratio.z;
    }

    update_tetrahedra_info_from_tetramesh(this->h_tetrahedra_info_1,
                                          this->m_0->t,
                                          nodes,
                                          true);

    //vec3 volume_out_position = this->world_to_device_scale.scale(tetramesh_aabb_1.aabb_min);


    vector<aabb_info> & my_aabbs = this->h_aabbs_1;
    vector<culling_grid_node> & all_grid_nodes = this->h_culling_grid_nodes;
    vector<uvec8> & all_grid_boxes = this->h_culling_grid_boxes;

    //Adapting cube and tetrahedron_info to the new deformation of tetrahedra mesh.
    this->create_tetra_aabb_grid(this->h_tetrahedra_info_1,
                                 mini_aabbs_dims,
                                 all_grid_nodes,
                                 all_grid_boxes);

    //------------- Computing nodes masks -------------
    static vector<unsigned char> nodes_masks;
    this->compute_grid_nodes_masks(this->h_tetrahedra_info_1,
                                   all_grid_nodes,
                                   nodes_masks);

    //------------- Computing boxes masks -------------
    static vector<bool> boxes_flags;
    this->compute_grid_boxes_masks(nodes_masks,
                                   all_grid_boxes,
                                   boxes_flags);

    //------------- Creating list of candidate boxes -------------
    this->create_candidate_aabb_info(all_grid_nodes,
                                     all_grid_boxes,
                                     boxes_flags,
                                     my_aabbs,
                                     mini_aabbs_dims);

    //------------- Rasterizing -------------
    VOXELTYPE clear_color = 0x00;
    this->cuda_rasterize(mini_aabbs_dims,
                         clear_color,
                         volume_out);

} //void voxelDeformer::deform(const vector<vec3> & nodes_1, voxelVolume & volume_out)






void voxelDeformer::deform_tex3d(const vector<vec3> & nodes_1, voxelVolume & volume_out) {

    if (this->vol_0 == NULL || this->vol_0 == NULL) {
        cerr << "ERROR: voxelDeformer::deform --> voxelDeformer not initialized, call init before!" << endl;
        return;
    }

    //Computing the size of the new voxel cube.
    this->tetramesh_aabb_1.set(nodes_1);
    vec3 ratio(tetramesh_aabb_1.width() / this->tetramesh_aabb_0.width(),
               tetramesh_aabb_1.height() / this->tetramesh_aabb_0.height(),
               tetramesh_aabb_1.depth() / this->tetramesh_aabb_0.depth());

    //    cout << "old tetramesh aabb " << this->tetramesh_aabb_0.to_s() << endl;
    //    cout << "new tetramesh aabb " << tetramesh_aabb_1.to_s() << endl;
    //    cout << "ratio " << ratio.to_s() << endl;

    this->volume_aabb_1.aabb_min.set(0,0,0);
    this->volume_aabb_1.aabb_max.set(int(ceil(this->volume_aabb_0.aabb_max.x * ratio.x)),
                                     int(ceil(this->volume_aabb_0.aabb_max.y * ratio.y)),
                                     int(ceil(this->volume_aabb_0.aabb_max.z * ratio.z)));

    //cout << "old volume aabb " << this->volume_aabb_0.to_s() << endl;
    //cout << "new volume aabb " << volume_aabb_1.to_s() << endl << endl;

    volume_out.set_dimensions(this->volume_aabb_1.aabb_max.x,
                              this->volume_aabb_1.aabb_max.y,
                              this->volume_aabb_1.aabb_max.z);

    createCudaOutputTexture3D(volume_out.width(),
                              volume_out.height(),
                              volume_out.depth());

//TODO: remove this method
    volume_out.resize(this->volume_aabb_1.aabb_max.x,
                      this->volume_aabb_1.aabb_max.y,
                      this->volume_aabb_1.aabb_max.z);


    //Typical cuda block of 512, which is 8x8x8.
    uvec3 mini_aabbs_dims = uvec3(8, 8, 8);


    //---------------- Convert all the nodes positions to voxel space ----------------

    //Convert all the nodes positions to voxel space.
    vector<vec3> nodes;
    nodes.resize(nodes_1.size());

    ratio.set(this->volume_aabb_1.width()  / tetramesh_aabb_1.width(),
              this->volume_aabb_1.height() / tetramesh_aabb_1.height(),
              this->volume_aabb_1.depth()  / tetramesh_aabb_1.depth());

    for (unsigned i=0; i<nodes.size(); i++) {
        const vec3 & n0 = nodes_1[i];
        vec3 & n = nodes[i];

        n.x = (n0.x - tetramesh_aabb_1.aabb_min.x) * ratio.x;
        n.y = (n0.y - tetramesh_aabb_1.aabb_min.y) * ratio.y;
        n.z = (n0.z - tetramesh_aabb_1.aabb_min.z) * ratio.z;
    }

    update_tetrahedra_info_from_tetramesh(this->h_tetrahedra_info_1,
                                          this->m_0->t,
                                          nodes,
                                          true);

    //vec3 volume_out_position = this->world_to_device_scale.scale(tetramesh_aabb_1.aabb_min);


    vector<aabb_info> & my_aabbs = this->h_aabbs_1;
    vector<culling_grid_node> & all_grid_nodes = this->h_culling_grid_nodes;
    vector<uvec8> & all_grid_boxes = this->h_culling_grid_boxes;

    //Adapting cube and tetrahedron_info to the new deformation of tetrahedra mesh.
    this->create_tetra_aabb_grid(this->h_tetrahedra_info_1,
                                 mini_aabbs_dims,
                                 all_grid_nodes,
                                 all_grid_boxes);

    //------------- Computing nodes masks -------------
    static vector<unsigned char> nodes_masks;
    this->compute_grid_nodes_masks(this->h_tetrahedra_info_1,
                                   all_grid_nodes,
                                   nodes_masks);

    //------------- Computing boxes masks -------------
    static vector<bool> boxes_flags;
    this->compute_grid_boxes_masks(nodes_masks,
                                   all_grid_boxes,
                                   boxes_flags);

    //------------- Creating list of candidate boxes -------------
    this->create_candidate_aabb_info(all_grid_nodes,
                                     all_grid_boxes,
                                     boxes_flags,
                                     my_aabbs,
                                     mini_aabbs_dims);

    //------------- Rasterizing -------------
    VOXELTYPE clear_color = 0x00;
    this->cuda_rasterize_tex3D(mini_aabbs_dims,
                               clear_color,
                               volume_out);

    //TODO: Put here.
    //destroyCudaOutputTexture3D();
} //void voxelDeformer::deform_tex3d(const vector<vec3> & nodes_1, voxelVolume & volume_out)




void voxelDeformer::finish() {
    this->vol_0 = NULL;
    this->volume_aabb_0.clear();
    this->volume_aabb_1.clear();
    this->m_0 = NULL;
    this->tetramesh_aabb_0.clear();

    //-------------- finish cuda rasterizer --------------

    cudaError cuda_result;

    //Finishing constant values.
    unLoadCudaInputTexture3D();
    destroyCudaOutputTexture3D();

    this->h_tetrahedra_info_0.clear();

    cuda_result = cudaFree(d_tetrahedra_info_0);
    if (cuda_result != cudaSuccess) {
        cerr << "ERROR CUDA: d_tetrahedra_info_0 cuda free failed\n";
        return;
    }
} //void voxelDeformer::finish()




void voxelDeformer::cuda_rasterize (const uvec3 mini_aabbs_dims,
                                    const VOXELTYPE clear_color,
                                    voxelVolume & volume_out) {
    cudaError cuda_result;

    //Repair the start block offset of each aabb.
    aux_repair_start_block_offsets(this->h_aabbs_1, mini_aabbs_dims);

    //Update the block2aabb lookup table:
    aux_update_block2aabb_lookup_table(this->h_aabbs_1, mini_aabbs_dims);

    //Uploading tetrahedra info to GPU.
    cuda_result = cudaMalloc((void**) &d_tetrahedra_info_1,
                             this->h_tetrahedra_info_1.size() * sizeof(tetrahedron_info));
    if (cuda_result != cudaSuccess) {
        cerr << "ERROR CUDA: cuda_upload_tetrahedra_info cuda malloc failed" << endl;
        return; //exit(-1);
    }
    cuda_result = cudaMemcpyAsync(d_tetrahedra_info_1,
                                  (void *) this->h_tetrahedra_info_1.data(),
                                  this->h_tetrahedra_info_1.size() * sizeof(tetrahedron_info),
                                  cudaMemcpyHostToDevice);
    if (cuda_result != cudaSuccess) {
        cerr << "ERROR CUDA: cuda_upload_tetrahedra_info cuda memcopy failed" << endl;
        return; //exit(-1);
    }


    //Upload h_aabbs_1 to CUDA:
    cuda_result = cudaMalloc((void**) &d_aabbs_1,
                             h_aabbs_1.size() * sizeof(aabb_info));
    if (cuda_result != cudaSuccess) {
        cerr << "ERROR CUDA: h_aabbs_1 cuda malloc failed\n";
        return; //exit(-1);
    }
    cuda_result = cudaMemcpyAsync(d_aabbs_1,
                                  (void *) h_aabbs_1.data(),
                                  h_aabbs_1.size() * sizeof(aabb_info),
                                  cudaMemcpyHostToDevice);
    if (cuda_result != cudaSuccess) {
        cerr << "ERROR CUDA: h_aabbs_1 cuda memcopy failed\n";
        return; //exit(-1);
    }

    cuda_result = cudaMalloc((void**) &d_cube_1, volume_out.num_bytes());
    if (cuda_result != cudaSuccess) {
        cerr << "ERROR CUDA: d_cube_1 cuda malloc failed, size ["<< volume_out.num_bytes() <<"] bytes\n";
        return;
    }

    //Setting deformed cube with all values to default transparent value
    cuda_result = cudaMemsetAsync(d_cube_1, clear_color, volume_out.num_bytes());

    if (cuda_result != cudaSuccess) {
        cerr << "ERROR CUDA: d_cube_1 cuda memset failed, size ["<< volume_out.num_bytes() <<"] bytes\n";
        return;
    }

    //------------------------- Calling CUDA method -------------------------

    int3 volume_dims;
    volume_dims.x = this->volume_aabb_1.width();
    volume_dims.y = this->volume_aabb_1.height();
    volume_dims.z = this->volume_aabb_1.depth();

    cuda_rasterize3DCall(h_block2aabb_lookup_table.size(),
                         d_tetrahedra_info_0,
                         d_block2aabb_lookup_table,
                         d_cube_1,
                         volume_dims,
                         d_tetrahedra_info_1,
                         d_aabbs_1);

    cuda_result = cudaGetLastError();
    if (cuda_result != cudaSuccess) {
        const char * error_string = cudaGetErrorString(cuda_result);
        cerr << "ERROR CUDA: Kernel execution failed, error code: ["<< cuda_result << "] text [" << error_string <<"]." << endl;
        return;
    }

    //Downloading results.
    h_cube_1 = volume_out.texels().data();
    cuda_result = cudaMemcpy(h_cube_1, d_cube_1, volume_out.num_bytes(), cudaMemcpyDeviceToHost);
    if (cuda_result != cudaSuccess) {
        cerr << "ERROR CUDA: h_cube_1 cuda memcopy failed, size ["<< volume_out.num_bytes() <<"] bytes\n";
        return;
    }

    cuda_result = cudaFree(d_block2aabb_lookup_table);
    if (cuda_result != cudaSuccess) {
        cerr << "ERROR CUDA: d_block2aabb_lookup_table cuda free failed\n";
        return;
    }

    cuda_result = cudaFree(d_aabbs_1);
    if (cuda_result != cudaSuccess) {
        cerr << "ERROR CUDA: d_aabbs_1 cuda free failed\n";
        return;
    }

    cuda_result = cudaFree(d_cube_1);
    if (cuda_result != cudaSuccess) {
        cerr << "ERROR CUDA: d_cube_1 cuda free failed\n";
        return;
    }

    cuda_result = cudaFree(d_tetrahedra_info_1);
    if (cuda_result != cudaSuccess) {
        cerr << "ERROR CUDA: d_tetrahedra_info_1 cuda free failed\n";
        return;
    }
} //voxelDeformer::cuda_rasterize()






void voxelDeformer::cuda_rasterize_tex3D (const uvec3 mini_aabbs_dims,
                                          const VOXELTYPE clear_color,
                                          voxelVolume & volume_out) {
    cudaError cuda_result;

    //Repair the start block offset of each aabb.
    aux_repair_start_block_offsets(this->h_aabbs_1, mini_aabbs_dims);

    //Update the block2aabb lookup table:
    aux_update_block2aabb_lookup_table(this->h_aabbs_1, mini_aabbs_dims);

    //Uploading tetrahedra info to GPU.
    cuda_result = cudaMalloc((void**) &d_tetrahedra_info_1,
                             this->h_tetrahedra_info_1.size() * sizeof(tetrahedron_info));
    if (cuda_result != cudaSuccess) {
        cerr << "ERROR CUDA: cuda_upload_tetrahedra_info cuda malloc failed" << endl;
        return; //exit(-1);
    }
    cuda_result = cudaMemcpyAsync(d_tetrahedra_info_1,
                                  (void *) this->h_tetrahedra_info_1.data(),
                                  this->h_tetrahedra_info_1.size() * sizeof(tetrahedron_info),
                                  cudaMemcpyHostToDevice);
    if (cuda_result != cudaSuccess) {
        cerr << "ERROR CUDA: cuda_upload_tetrahedra_info cuda memcopy failed" << endl;
        return; //exit(-1);
    }


    //Upload h_aabbs_1 to CUDA:
    cuda_result = cudaMalloc((void**) &d_aabbs_1,
                             h_aabbs_1.size() * sizeof(aabb_info));
    if (cuda_result != cudaSuccess) {
        cerr << "ERROR CUDA: h_aabbs_1 cuda malloc failed\n";
        return; //exit(-1);
    }
    cuda_result = cudaMemcpyAsync(d_aabbs_1,
                                  (void *) h_aabbs_1.data(),
                                  h_aabbs_1.size() * sizeof(aabb_info),
                                  cudaMemcpyHostToDevice);
    if (cuda_result != cudaSuccess) {
        cerr << "ERROR CUDA: h_aabbs_1 cuda memcopy failed\n";
        return; //exit(-1);
    }

    cuda_result = cudaMalloc((void**) &d_cube_1, volume_out.num_bytes());
    if (cuda_result != cudaSuccess) {
        cerr << "ERROR CUDA: d_cube_1 cuda malloc failed, size ["<< volume_out.num_bytes() <<"] bytes\n";
        return;
    }

    //Setting deformed cube with all values to default transparent value
    cuda_result = cudaMemsetAsync(d_cube_1, clear_color, volume_out.num_bytes());

    if (cuda_result != cudaSuccess) {
        cerr << "ERROR CUDA: d_cube_1 cuda memset failed, size ["<< volume_out.num_bytes() <<"] bytes\n";
        return;
    }

    //--------------------- Creating texture for writting -------------------
/*
    //Creating a 3D texture.
    GLuint tex_id = load_texture_3d(this->volume_aabb_1.width(),
                                    this->volume_aabb_1.height(),
                                    this->volume_aabb_1.depth());

    //Registering texture as a Cuda image.
    struct cudaGraphicsResource * cuda_image_resource;

    cuda_result = cudaGraphicsGLRegisterImage(&cuda_image_resource,
                                              tex_id,
                                              GL_TEXTURE_3D,
                                              cudaGraphicsRegisterFlagsSurfaceLoadStore);
    if (cuda_result != cudaSuccess) {
        cerr << "ERROR CUDA: cudaGraphicsGLRegisterImage failed" << endl;
        return; //exit(-1);
    }

    cuda_result = cudaGraphicsMapResources(1, &cuda_image_resource, 0);
    if (cuda_result != cudaSuccess) {
        cerr << "ERROR CUDA: cudaGraphicsMapResources failed" << endl;
        return; //exit(-1);
    }

    cudaArray * cuda_image_array;

    cuda_result = cudaGraphicsSubResourceGetMappedArray(&cuda_image_array,
                                                        cuda_image_resource,
                                                        0, 0);
    if (cuda_result != cudaSuccess) {
        cerr << "ERROR CUDA: cudaGraphicsSubResourceGetMappedArray failed" << endl;
        return; //exit(-1);
    }

*/

    //------------------------- Calling CUDA method -------------------------

    int3 volume_dims;
    volume_dims.x = this->volume_aabb_1.width();
    volume_dims.y = this->volume_aabb_1.height();
    volume_dims.z = this->volume_aabb_1.depth();

    cuda_rasterize_tex3DCall(h_block2aabb_lookup_table.size(),
                             d_tetrahedra_info_0,
                             d_block2aabb_lookup_table,
                             d_cube_1,
                             //cuda_image_array,
                             volume_dims,
                             d_tetrahedra_info_1,
                             d_aabbs_1);

    cudaDeviceSynchronize();

    cuda_result = cudaGetLastError();
    if (cuda_result != cudaSuccess) {
        const char * error_string = cudaGetErrorString(cuda_result);
        cerr << "ERROR CUDA: Kernel execution failed, error code: ["<< cuda_result << "] text [" << error_string <<"]." << endl;
        return;
    }

    //Downloading results.
    h_cube_1 = volume_out.texels().data();
    cuda_result = cudaMemcpy(h_cube_1, d_cube_1, volume_out.num_bytes(), cudaMemcpyDeviceToHost);
    if (cuda_result != cudaSuccess) {
        cerr << "ERROR CUDA: h_cube_1 cuda memcopy failed, size ["<< volume_out.num_bytes() <<"] bytes\n";
        return;
    }

    cuda_result = cudaFree(d_block2aabb_lookup_table);
    if (cuda_result != cudaSuccess) {
        cerr << "ERROR CUDA: d_block2aabb_lookup_table cuda free failed\n";
        return;
    }

    cuda_result = cudaFree(d_aabbs_1);
    if (cuda_result != cudaSuccess) {
        cerr << "ERROR CUDA: d_aabbs_1 cuda free failed\n";
        return;
    }

    cuda_result = cudaFree(d_cube_1);
    if (cuda_result != cudaSuccess) {
        cerr << "ERROR CUDA: d_cube_1 cuda free failed\n";
        return;
    }

    cuda_result = cudaFree(d_tetrahedra_info_1);
    if (cuda_result != cudaSuccess) {
        cerr << "ERROR CUDA: d_tetrahedra_info_1 cuda free failed\n";
        return;
    }
/*
    //Finishing texture writting resources.
    cuda_result = cudaGraphicsUnmapResources(1, &cuda_image_resource, 0);
    if (cuda_result != cudaSuccess) {
        cerr << "ERROR CUDA: cudaGraphicsUnmapResources failed" << endl;
        return; //exit(-1);
    }

    cuda_result = cudaGraphicsUnregisterResource(cuda_image_resource);
    if (cuda_result != cudaSuccess) {
        cerr << "ERROR CUDA: cudaGraphicsUnregisterResource failed" << endl;
        return; //exit(-1);
    }
*/
    //unload_texture_3d(tex_id);
//    this->vol_0->tex3D_id() = tex_id;

} //voxelDeformer::cuda_rasterize_tex3D()





void voxelDeformer::aux_repair_start_block_offsets(vector<aabb_info> & aabbs,
                                                       const uvec3 aabb_dims) {
    unsigned accumulated_start_offset = 0;
    for (unsigned aabb_i=0; aabb_i < aabbs.size(); aabb_i++) {
        aabb_info & bb = aabbs[aabb_i];
        bb.start_block_offset = accumulated_start_offset;
        accumulated_start_offset += int(ceil(float(aabb_dims.x * aabb_dims.y * aabb_dims.z) / float(THREADS_PER_BLOCK)));
    }
}



void voxelDeformer::aux_update_block2aabb_lookup_table (vector<aabb_info> & aabbs, const uvec3 aabb_dims) {

//    cout << "Populating block2aabb_lookup_table --> " << aabbs.size() << " aabbs" << endl;
    //Build the "block-->aabb" lookup table.
    h_block2aabb_lookup_table.clear();
    h_block2aabb_lookup_table.reserve(aabbs.size());
    unsigned number_aabb_blocks;
    unsigned cell_size;

    //Populating the "block-->aabb" lookup table.
    for (unsigned aabb_i=0; aabb_i < aabbs.size(); aabb_i++) {
        //aabb_info & bb = aabbs[aabb_i];
        cell_size = aabb_dims.x * aabb_dims.y * aabb_dims.z;
        number_aabb_blocks = cell_size / THREADS_PER_BLOCK;
        if (cell_size > number_aabb_blocks * THREADS_PER_BLOCK) {
            number_aabb_blocks++;
        }
        for (unsigned block_i=0; block_i < number_aabb_blocks; block_i++) {
            h_block2aabb_lookup_table.push_back(aabb_i);
        }
    }

    //Upload h_block2aabb_lookup_table to CUDA:
    cudaError cuda_result = cudaMalloc((void**) &d_block2aabb_lookup_table,
                                       h_block2aabb_lookup_table.size() * sizeof(unsigned));
    if (cuda_result != cudaSuccess) {
        cerr << "ERROR CUDA: h_block2aabb_lookup_table cuda malloc failed\n";
        return; //exit(-1);
    }
    cuda_result = cudaMemcpyAsync(d_block2aabb_lookup_table,
                             (void *) h_block2aabb_lookup_table.data(),
                             h_block2aabb_lookup_table.size() * sizeof(unsigned),
                             cudaMemcpyHostToDevice);
    if (cuda_result != cudaSuccess) {
        cerr << "ERROR CUDA: h_block2aabb_lookup_table cuda memcopy failed\n";
        return; //exit(-1);
    }
}



unsigned char voxelDeformer::aux_set_barycentrics_mask(const vec3 & bar) const {
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




void voxelDeformer::create_tetra_aabb_grid(const vector<tetrahedron_info> & tetras,
                                           const uvec3 new_aabbs_dims,
                                           vector<culling_grid_node> & result_grid_nodes,
                                           vector<uvec8> & result_grid_boxes) {
    uvec3 steps;

    unsigned total_number_nodes = 0;
    unsigned total_number_boxes = 0;

    static vector<unsigned> my_start_nodes_index(tetras.size() * 2);
    static vector<unsigned> my_start_boxes_index(tetras.size() * 2);

    my_start_nodes_index.resize(tetras.size());
    my_start_boxes_index.resize(tetras.size());

    //Estimating the total number of grid nodes and boxes.
    for (unsigned tetra_i=0; tetra_i<tetras.size(); tetra_i++) {
        const tetrahedron_info & tetra = tetras[tetra_i];
        steps.x = CEIL_POS(float(tetra.dims.x) / float(new_aabbs_dims.x));
        steps.y = CEIL_POS(float(tetra.dims.y) / float(new_aabbs_dims.y));
        steps.z = CEIL_POS(float(tetra.dims.z) / float(new_aabbs_dims.z));

        my_start_nodes_index[tetra_i] = total_number_nodes;
        my_start_boxes_index[tetra_i] = total_number_boxes;

        total_number_nodes += (steps.x + 1) * (steps.y + 1) * (steps.z + 1);
        total_number_boxes += steps.x * steps.y * steps.z;
    }

    result_grid_nodes.resize(total_number_nodes);
    result_grid_boxes.resize(total_number_boxes);

    culling_grid_node current_node;

    #pragma omp parallel for num_threads(omp_get_num_procs())
    for (unsigned tetra_i=0; tetra_i<tetras.size(); tetra_i++) {
        const tetrahedron_info & tetra = tetras[tetra_i];
        iaabb current_iaabb;
        current_iaabb.aabb_min.x = tetra.iaabb_min.x;
        current_iaabb.aabb_min.y = tetra.iaabb_min.y;
        current_iaabb.aabb_min.z = tetra.iaabb_min.z;

        current_iaabb.aabb_max.x = tetra.iaabb_min.x + tetra.dims.x;
        current_iaabb.aabb_max.y = tetra.iaabb_min.y + tetra.dims.y;
        current_iaabb.aabb_max.z = tetra.iaabb_min.z + tetra.dims.z;

        create_ibox_labeled_grid(current_iaabb,
                                 new_aabbs_dims,
                                 result_grid_nodes,
                                 my_start_nodes_index[tetra_i],
                                 //start_grid_nodes_index,
                                 result_grid_boxes,
                                 //start_grid_boxes_index,
                                 my_start_boxes_index[tetra_i],
                                 tetra_i);
    }
}






void voxelDeformer::compute_grid_nodes_masks(const vector<tetrahedron_info> & tetras,
                                                 const vector<culling_grid_node> & grid_nodes,
                                                 vector<unsigned char> & result_nodes_masks) {
    result_nodes_masks.resize(grid_nodes.size(), 0x00);
    #pragma omp parallel for num_threads(omp_get_num_procs())
    for (unsigned i=0; i<grid_nodes.size(); i++) {
        const culling_grid_node & n = grid_nodes[i];
        const tetrahedron_info & t = tetras[n.tetra_i];
        const vec3 p(n.node.x - t.node_0.x,
                     n.node.y - t.node_0.y,
                     n.node.z - t.node_0.z);
        vec3 bar;
        fmul3x3(bar, t.bar_mat, p);
        result_nodes_masks[i] = aux_set_barycentrics_mask(bar);
    }
}




void voxelDeformer::compute_grid_boxes_masks(const vector<unsigned char> & grid_nodes_masks,
                                                 const vector<uvec8> & grid_boxes,
                                                 vector<bool> & result_boxes_flags) {
    result_boxes_flags.resize(grid_boxes.size());

    #pragma omp parallel for num_threads(omp_get_num_procs())
    for (unsigned i=0; i<grid_boxes.size(); i++) {
        const uvec8 & box = grid_boxes[i];
        unsigned char box_mask = grid_nodes_masks[box[0]];
        for (unsigned j=1; j<8 && box_mask != 0x00; j++) {
            box_mask &= grid_nodes_masks[box[j]];
        }
        //True means "culled".
        result_boxes_flags[i] = box_mask;
    }
}




void voxelDeformer::create_all_aabb_info(const vector<culling_grid_node> & grid_nodes,
                                             const vector<uvec8> & grid_boxes,
                                             vector<aabb_info> & result_aabbs,
                                             const uvec3 aabb_dims,
                                             unsigned threads_per_block) {
    iaabb bounds;
    result_aabbs.resize(grid_boxes.size());
    unsigned current_offset = 0;

    for (unsigned box_i=0; box_i<grid_boxes.size(); box_i++) {
        const ivec3 p = grid_nodes[grid_boxes[box_i][0]].node;

        aabb_info & cell = result_aabbs[box_i];

        cell.iaabb_min.x = p.x;
        cell.iaabb_min.y = p.y;
        cell.iaabb_min.z = p.z;

        cell.index = grid_nodes[grid_boxes[box_i][0]].tetra_i;
        cell.start_block_offset = current_offset;
        current_offset += unsigned(ceil(float(aabb_dims.x * aabb_dims.y * aabb_dims.z) / float(threads_per_block)));
    }
}




void voxelDeformer::create_candidate_aabb_info(const vector<culling_grid_node> & grid_nodes,
                                                   const vector<uvec8> & grid_boxes,
                                                   const vector<bool> & boxes_mask,
                                                   vector<aabb_info> & result_aabbs,
                                                   const uvec3 aabb_dims,
                                                   unsigned threads_per_block) {
    iaabb bounds;
    result_aabbs.resize(0);
    result_aabbs.reserve(grid_boxes.size());
    aabb_info cell;
    unsigned current_offset = 0;
    unsigned aabb_blocks = 0;

    for (unsigned box_i=0; box_i<grid_boxes.size(); box_i++) {
        //If the box is worth to be rendered (true or != 0 means "culled")
        if (boxes_mask[box_i] == false) {
            const ivec3 p = grid_nodes[grid_boxes[box_i][0]].node;

            cell.iaabb_min.x = p.x;
            cell.iaabb_min.y = p.y;
            cell.iaabb_min.z = p.z;

            cell.index = grid_nodes[grid_boxes[box_i][0]].tetra_i;
            cell.grid_box_i = box_i;

            cell.start_block_offset = current_offset;
            aabb_blocks = unsigned(ceil(float(aabb_dims.x * aabb_dims.y * aabb_dims.z) / float(threads_per_block)));
            current_offset += aabb_blocks;
            result_aabbs.push_back(cell);
        }
    }
}


