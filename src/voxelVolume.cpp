
#include <voxelVolume.hpp>

void paint_voxel_aabb_edges(vector<unsigned char> & texels,
                            const ivec3 & aabb_min,
                            const ivec3 & aabb_max,
                            const uvec3 & dimensions,
                            const unsigned char color) {

    for (int i=aabb_min.x; i<=aabb_max.x; i++) {
        texels[offset(i, aabb_min.y, aabb_min.z, dimensions)] = color;
        texels[offset(i, aabb_min.y, aabb_max.z, dimensions)] = color;
        texels[offset(i, aabb_max.y, aabb_min.z, dimensions)] = color;
        texels[offset(i, aabb_max.y, aabb_max.z, dimensions)] = color;
    }
    for (int j=aabb_min.y; j<=aabb_max.y; j++) {
        texels[offset(aabb_min.x, j, aabb_min.z, dimensions)] = color;
        texels[offset(aabb_min.x, j, aabb_max.z, dimensions)] = color;
        texels[offset(aabb_max.x, j, aabb_min.z, dimensions)] = color;
        texels[offset(aabb_max.x, j, aabb_max.z, dimensions)] = color;
    }
    for (int k=aabb_min.z; k<=aabb_max.z; k++) {
        texels[offset(aabb_min.x, aabb_min.y, k, dimensions)] = color;
        texels[offset(aabb_min.x, aabb_max.y, k, dimensions)] = color;
        texels[offset(aabb_max.x, aabb_min.y, k, dimensions)] = color;
        texels[offset(aabb_max.x, aabb_max.y, k, dimensions)] = color;
    }
}



voxelVolume::voxelVolume() {
    this->clear();
}



voxelVolume::~voxelVolume() {
    this->clear();
}



bool voxelVolume::load_raw(string filename, unsigned width, unsigned height, unsigned depth) {

    this->resize(width, height, depth);
    FILE *f = fopen(filename.c_str(), "rb");
    if (!f) {
        cerr << "Error: voxelVolume::load_raw --> file ["<< filename <<"] could not be opened" << endl;
        throw filename+string(": File not found.");
        return false;
    }
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    cout << "    Reading img file ["<< filename <<"] of " << file_size << " bytes." << endl;

    long estimated_size = width * height * depth * sizeof(VOXELTYPE);

    if (file_size != estimated_size) {
        cerr << "Error: load_voxel_raw --> filename ["<< filename <<"] file size is different to expected,";
        cerr << " estimated ["<< estimated_size <<"] real ["<< file_size <<"]." << endl;
        throw filename+string(": File size different from expected.");
        return false;
    }

    unsigned char * tempBuff = new VOXELTYPE[estimated_size];
    if (fread(tempBuff, estimated_size, 1, f) != 1) {
        cerr << "Error: load_voxel_raw --> filename ["<< filename <<"] failed reading file." << endl;
        delete tempBuff;
        throw filename+string(": Error while reading file.");
        return false;
    }

    for (unsigned i=0; i<this->_texels.size(); i++) {
        this->_texels[i] = tempBuff[i];
    }
    delete tempBuff;
    return true;
} //bool voxelVolume::load_raw(string filename, unsigned width, unsigned height, unsigned depth)





bool voxelVolume::load(string filename, unsigned width, unsigned height, unsigned depth) {

    string ext = filename.substr(filename.find_last_of(".")+1);
    //cout << "Loading filename [" << filename << "] with extension ["<< ext << "]" << endl;

    if (ext == "raw") {
        if (width == 0 || width == 0 || width == 0) {
            cerr << "ERROR: voxelVolume::load --> For loading a .raw file I need the width, height and depth!" << endl;
            return false;
        }
        return this->load_raw(filename, width, height, depth);
    }

    //TODO: Add support for more formats, as PVM, DICOM and so on.

    cerr << "ERROR: voxelVolume::load --> Unknown loader for this file ["<< filename <<"]" << endl;

    //TODO: Loading actual filename of the volumetric data.

    //For now, generating a small cube.
    //this->resize(40, 28, 15, 0x00);
    this->resize(20, 18, 21, 0x00);
    //this->resize(255, 255, 255, 0x00);

    ivec3 aabb_min(0, 0, 0);
    ivec3 aabb_max(this->_dimensions.x-1, this->_dimensions.y-1, this->_dimensions.z-1);

    //Painting the edges with white voxels.
    paint_voxel_aabb_edges(this->_texels,
                           aabb_min,
                           aabb_max,
                           this->_dimensions,
                           0xFF);
    return true;
} //voxelVolume::load(string filename, unsigned width, unsigned height, unsigned depth)



void voxelVolume::resize(unsigned width,
                         unsigned height,
                         unsigned depth,
                         unsigned char default_value) {
    this->_dimensions.set(width, height, depth);
    this->_texels.resize(this->_dimensions.x * this->_dimensions.y * this->_dimensions.z,
                         default_value);
} //void voxelVolume::resize(unsigned width,...



void voxelVolume::set_dimensions(unsigned width,
                                 unsigned height,
                                 unsigned depth) {
    this->_dimensions.set(width, height, depth);
} //void voxelVolume::set_dimensions(unsigned width,...



void voxelVolume::clear() {
    this->_texels.clear();
    this->_dimensions.set(0,0,0);
    this->_tex3D_id = 0;
} //void voxelVolume::clear()
