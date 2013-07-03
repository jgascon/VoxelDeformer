#include <opengl_draw_methods.hpp>
#include <shaders.hpp>


GLuint load_texture_3d(unsigned width,
                       unsigned height,
                       unsigned depth,
                       unsigned char * texels) {

    //cout << "load_texture_3d --> loading texture of " << width << "x" << height << "x" << depth << endl;
    glEnable(GL_TEXTURE_3D);

    GLuint tex_id = 0;
    glGenTextures(1, &tex_id);
    glBindTexture(GL_TEXTURE_3D, tex_id);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    //glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    //glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    //glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    //glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    //glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

    glTexImage3D(GL_TEXTURE_3D,
                 0,
                 GL_LUMINANCE,
                 width,
                 height,
                 depth,
                 0,
                 GL_LUMINANCE,
                 GL_UNSIGNED_BYTE,
                 texels);

    GLenum error_code = glGetError();
    if (error_code != GL_NO_ERROR) {
        const GLubyte * error_text = gluErrorString(error_code);
        cerr << "ERROR: load_texture_3d --> code: [" << error_code << "] text: [" << error_text <<"]" << endl;
    }
    glBindTexture(GL_TEXTURE_3D, 0);
    glDisable(GL_TEXTURE_3D);
    return tex_id;
}



void unload_texture_3d(GLuint & tex_id) {
    if (tex_id != 0) {
        glDeleteTextures(1, &tex_id);
        tex_id = 0;
        GLenum error_code = glGetError();
        if (error_code != GL_NO_ERROR) {
            const GLubyte * error_text = gluErrorString(error_code);
            cerr << "ERROR: unload_texture_3d --> code: [" << error_code << "] text: [" << error_text <<"]" << endl;
        }
    }
}



void select_texture_3d(GLuint tex_id, unsigned texture_unit_id) {
    glActiveTexture(GL_TEXTURE0 + texture_unit_id);
    glEnable(GL_TEXTURE_3D);
    glBindTexture(GL_TEXTURE_3D, tex_id);
}



void deselect_texture_3d(unsigned texture_unit_id) {
    glActiveTexture(GL_TEXTURE0 + texture_unit_id);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_3D);
}


void draw_solid_aabb(const aabb & bb) {

    glBegin(GL_QUADS);
        //First quad.
        glVertex3d(bb.aabb_min.x, bb.aabb_min.y, bb.aabb_min.z);
        glVertex3d(bb.aabb_max.x, bb.aabb_min.y, bb.aabb_min.z);
        glVertex3d(bb.aabb_max.x, bb.aabb_max.y, bb.aabb_min.z);
        glVertex3d(bb.aabb_min.x, bb.aabb_max.y, bb.aabb_min.z);

        //Second quad.
        glVertex3d(bb.aabb_min.x, bb.aabb_min.y, bb.aabb_max.z);
        glVertex3d(bb.aabb_min.x, bb.aabb_max.y, bb.aabb_max.z);
        glVertex3d(bb.aabb_max.x, bb.aabb_max.y, bb.aabb_max.z);
        glVertex3d(bb.aabb_max.x, bb.aabb_min.y, bb.aabb_max.z);


        //Third quad.
        glVertex3d(bb.aabb_min.x, bb.aabb_min.y, bb.aabb_min.z);
        glVertex3d(bb.aabb_min.x, bb.aabb_max.y, bb.aabb_min.z);
        glVertex3d(bb.aabb_min.x, bb.aabb_max.y, bb.aabb_max.z);
        glVertex3d(bb.aabb_min.x, bb.aabb_min.y, bb.aabb_max.z);

        //Fourth quad.
        glVertex3d(bb.aabb_max.x, bb.aabb_min.y, bb.aabb_min.z);
        glVertex3d(bb.aabb_max.x, bb.aabb_min.y, bb.aabb_max.z);
        glVertex3d(bb.aabb_max.x, bb.aabb_max.y, bb.aabb_max.z);
        glVertex3d(bb.aabb_max.x, bb.aabb_max.y, bb.aabb_min.z);


        //Fifth quad.
        glVertex3d(bb.aabb_min.x, bb.aabb_max.y, bb.aabb_min.z);
        glVertex3d(bb.aabb_max.x, bb.aabb_max.y, bb.aabb_min.z);
        glVertex3d(bb.aabb_max.x, bb.aabb_max.y, bb.aabb_max.z);
        glVertex3d(bb.aabb_min.x, bb.aabb_max.y, bb.aabb_max.z);

        //Sixth quad.
        glVertex3d(bb.aabb_min.x, bb.aabb_min.y, bb.aabb_min.z);
        glVertex3d(bb.aabb_min.x, bb.aabb_min.y, bb.aabb_max.z);
        glVertex3d(bb.aabb_max.x, bb.aabb_min.y, bb.aabb_max.z);
        glVertex3d(bb.aabb_max.x, bb.aabb_min.y, bb.aabb_min.z);
    glEnd();
}



void draw_edges(const vector<uvec2> & e,
                const vector<vec3> & v,
                float x_offset,
                float y_offset,
                float z_offset) {
    //Painting edges.
    uvec2 ed;
    vec3 x, y;
    glBegin(GL_LINES);
        for(unsigned i=0; i < e.size(); i++) {
            ed = e[i];
            x = v[ed.x];
            glVertex3f(x.x+x_offset, x.y+y_offset, x.z+z_offset);
            y = v[ed.y];
            glVertex3f(y.x+x_offset, y.y+y_offset, y.z+z_offset);
        }
    glEnd();
}




void draw_nodes(const vector<vec3> & m, float x_offset, float y_offset, float z_offset) {
    glBegin(GL_POINTS);
    vec3 p;
    for(unsigned i=0; i < m.size(); i++) {
        p = m[i];
        glVertex3f(p.x+x_offset, p.y+y_offset, p.z+z_offset);
    }
    glEnd();
}







void draw_volume(const aabb & volume_bb,
                 unsigned tex3D_id,
                 unsigned shader_id) {

    glColor4f(1.0, 1.0, 1.0, 1.0);

    select_texture_3d(tex3D_id);

    Shaders::get_ref().enable_shader(shader_id);

    GLint shader_param = glGetUniformLocation(shader_id, "azimuth");
    glUniform1f(shader_param, -421.0);

    shader_param = glGetUniformLocation(shader_id, "elevation");
    glUniform1f(shader_param, 124.0);

    shader_param = glGetUniformLocation(shader_id, "clipPlaneDepth");
    glUniform1f(shader_param, 0.0);

    shader_param = glGetUniformLocation(shader_id, "clip");
    glUniform1i(shader_param, 0);

    shader_param = glGetUniformLocation(shader_id, "dither");
    glUniform1i(shader_param, 0);

    shader_param = glGetUniformLocation(shader_id, "tex_3d");
    glUniform1i(shader_param, 0);

    shader_param = glGetUniformLocation(shader_id, "aabb_min");
    glUniform3f(shader_param, volume_bb.aabb_min.x, volume_bb.aabb_min.y, volume_bb.aabb_min.z);

    shader_param = glGetUniformLocation(shader_id, "aabb_max");
    glUniform3f(shader_param, volume_bb.aabb_max.x, volume_bb.aabb_max.y, volume_bb.aabb_max.z);

    //Getting the AABB of the tetramesh and drawing it.
    draw_solid_aabb(volume_bb);

    Shaders::get_ref().disable_shader();

    deselect_texture_3d();
}


