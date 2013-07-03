
#include <types.hpp>


inline unsigned offset (const unsigned x, const unsigned y, const unsigned z, const uvec3 dimensions) {
    return x + dimensions.x * (y + dimensions.y * z);
}


void paint_voxel_aabb_edges(vector<unsigned char> & texels,
                            const ivec3 & aabb_min,
                            const ivec3 & aabb_max,
                            const uvec3 & dimensions,
                            const unsigned char color);


typedef unsigned char VOXELTYPE;




class voxelVolume {

    protected:

        ///Original data of the volume.
        vector<VOXELTYPE> _texels;

        ///(width, height, depth) of the volume.
        uvec3 _dimensions;

        ///Id of the texture 3D.
        unsigned _tex3D_id;

        ///Load a raw filename.
        bool load_raw(string filename, unsigned width, unsigned height, unsigned depth);

    public:

        voxelVolume();

        ~voxelVolume();

        ///Loads a filename.
        bool load(string filename,
                  unsigned width = 0,
                  unsigned height = 0,
                  unsigned depth = 0);

        void resize(unsigned width,
                    unsigned height,
                    unsigned depth,
                    unsigned char default_value = 0x00);

        void set_dimensions(unsigned width,
                            unsigned height,
                            unsigned depth);

        void clear();

        //------------------------- Accesors -------------------------

        vector<VOXELTYPE> & texels() { return this->_texels; }

        uvec3 dimensions() { return this->_dimensions; }

        unsigned width() { return this->_dimensions.x; }

        unsigned height() { return this->_dimensions.y; }

        unsigned depth() { return this->_dimensions.z; }

        unsigned num_voxels() const { return this->_dimensions.x * this->_dimensions.y * this->_dimensions.z; }

        unsigned num_bytes() const { return this->num_voxels() * sizeof(VOXELTYPE); }

        unsigned & tex3D_id() { return this->_tex3D_id; }
};
