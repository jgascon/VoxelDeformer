#ifndef OPENGL_DRAW_METHODS_HPP
#define OPENGL_DRAW_METHODS_HPP

#define GL_GLEXT_PROTOTYPES
#if defined (__APPLE__) || defined(MACOSX)
    #include <GLUT/glut.h>
#else
    #include <GL/freeglut.h>
#endif
//#include <GL/glut.h>
#include <types.hpp>
#include <shaders.hpp>

GLuint load_texture_3d(unsigned width,
                       unsigned height,
                       unsigned depth,
                       unsigned char * texels = NULL);

void unload_texture_3d(GLuint & tex_id);


void select_texture_3d(GLuint tex_id, unsigned texture_unit_id = 0);

void deselect_texture_3d(unsigned texture_unit_id = 0);

void draw_solid_aabb(const aabb & bb);

void draw_edges(const vector<uvec2> & e,
                const vector<vec3> & v,
                float x_offset = 0.0,
                float y_offset = 0.0,
                float z_offset = 0.0);


void draw_nodes(const vector<vec3> & m, float x_offset = 0.0, float y_offset = 0.0, float z_offset = 0.0);


void draw_volume(const aabb & volume_bb,
                 unsigned tex3D_id,
                 unsigned shader_id);

#endif // OPENGL_DRAW_METHODS_HPP
