#ifndef VOXELDEFORMER_HPP
#define VOXELDEFORMER_HPP

#include <types.hpp>
#include <matrix.hpp>
#include <voxelVolume.hpp>
#include <culler_rasterizer.cuh>
#include <opengl_draw_methods.hpp>
#include <omp.h>


class voxelDeformer {

    protected:

        ///Original voxel volume for this deformer.
        voxelVolume * vol_0;

        ///Original voxel volume aabb for this deformer.
        iaabb volume_aabb_0;

        ///Deformed voxel volume aabb for this deformer.
        iaabb volume_aabb_1;

        ///Original tetramesh for this deformer.
        tetramesh * m_0;

        ///Original tetramesh aabb for this deformer.
        aabb tetramesh_aabb_0;

        ///Deformed tetramesh aabb for this deformer.
        aabb tetramesh_aabb_1;

        //--------------- Rasterizer attributes ---------------

        //List of all tetrahedra info (barycentrix matrices, nodes_0, and so on).
        vector<tetrahedron_info> h_tetrahedra_info_0;
        tetrahedron_info * d_tetrahedra_info_0;

        vector<tetrahedron_info> h_tetrahedra_info_1;
        tetrahedron_info * d_tetrahedra_info_1;

        //Original AABB_SET, current one and culled one.
        vector<aabb_info> h_aabbs_1;
        aabb_info * d_aabbs_1;

        //List of masks for nodes and boxes.
        vector<unsigned char> h_nodes_masks;
        vector<bool> h_boxes_flags;

        //Lookup table "block_id --> aabb"
        vector<unsigned> h_block2aabb_lookup_table;
        unsigned * d_block2aabb_lookup_table;


        //List of nodes for the culling grid.
        vector<culling_grid_node> h_culling_grid_nodes;

        //List of boxes for the culling grid.
        vector<uvec8> h_culling_grid_boxes;


        //Voxels cube.
        VOXELTYPE * h_cube_0, * d_cube_0;
        VOXELTYPE * h_cube_1, * d_cube_1;

        //Voxels cube properties.
        uvec3 h_cube_dims_0;
        uvec3 h_cube_dims_1;


        //--------------- Rasterizer internal methods ---------------

        //Auxiliary methods for updating internal culler_rasterizer structures.
        void aux_repair_start_block_offsets (vector<aabb_info> & aabbs,
                                             const uvec3 aabb_dims = uvec3(8,8,8));

        void aux_update_block2aabb_lookup_table (vector<aabb_info> & aabbs,
                                                 const uvec3 aabb_dims = uvec3(8,8,8));

        unsigned char aux_set_barycentrics_mask(const vec3 & bar) const;


        //New culling methods
        void create_tetra_aabb_grid(const vector<tetrahedron_info> & tetras,
                                    const uvec3 new_aabbs_dims,
                                    vector<culling_grid_node> & result_grid_nodes,
                                    vector<uvec8> & result_grid_boxes);

        //Create grid nodes' masks.
        void compute_grid_nodes_masks(const vector<tetrahedron_info> & tetras,
                                      const vector<culling_grid_node> & grid_nodes,
                                      vector<unsigned char> & result_nodes_masks);

        //Create grid boxes' masks.
        void compute_grid_boxes_masks(const vector<unsigned char> & grid_nodes_masks,
                                      const vector<uvec8> & grid_boxes,
                                      vector<bool> & result_boxes_flags);

        //Create all "aabb_info" from the input grid:
        void create_all_aabb_info(const vector<culling_grid_node> & grid_nodes,
                                  const vector<uvec8> & grid_boxes,
                                  vector<aabb_info> & result_aabbs,
                                  const uvec3 aabb_dims,
                                  unsigned threads_per_block = 512);


        //Create "aabb_info" candidates for rasterizing:
        void create_candidate_aabb_info(const vector<culling_grid_node> & grid_nodes,
                                        const vector<uvec8> & grid_boxes,
                                        const vector<bool> &boxes_mask,
                                        vector<aabb_info> & result_aabbs,
                                        const uvec3 aabb_dims,
                                        unsigned threads_per_block = 512);

        void cuda_rasterize (const uvec3 mini_aabbs_dims,
                             const VOXELTYPE clear_color,
                             voxelVolume &volume_out);

        void cuda_rasterize_tex3D (const uvec3 mini_aabbs_dims,
                                   const VOXELTYPE clear_color,
                                   voxelVolume & volume_out);

    public:

        voxelDeformer();

        ~voxelDeformer();

        void init(voxelVolume * volume, tetramesh * tetra);

        void deform(const vector<vec3> & nodes_1, voxelVolume & volume_out);

        void deform_tex3d(const vector<vec3> & nodes_1, voxelVolume & volume_out);

        void finish();
};

#endif // VOXELDEFORMER_HPP
