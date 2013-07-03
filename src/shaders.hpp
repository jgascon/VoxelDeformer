//
// C++ Interface: shaders
//
// Description:
//
//
// Author: Jorge Gascon Perez,,, <jorge@IronMaiden>, (C) 2009
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef SHADERS_H
#define SHADERS_H

//#include <utilities/libGraphics.h>

#include <stdio.h>
#include <stdlib.h>

#define GL_GLEXT_PROTOTYPES
//#include <GL/gl.h>
#include <GL/glut.h>
//#include <GL/glu.h>
//#include <GL/glext.h>
//#include <GL/glew.h>
#include <string>
#include <map>
#include <iostream>

//#include <utilities/mesh.h>

using namespace std;

/**
    @brief This class represents a shader.
    Each shader has a Vertex shader, a Fragment Shader and optional a Geometry Shader.
    (Geometry Shader is not implemented here for now).
*/
typedef struct Shader {

    ///Shader program unique identifier.
    GLuint id;
    ///Vertex shader unique identifier.
    GLuint vertex_id;
    ///Fragment shader unique identifier.
    GLuint fragment_id;

    Shader() {
        id = 0;
        vertex_id = 0;
        fragment_id = 0;
    }
} Shader;



/**
     @author Jorge Gascon Perez,,, <jorge@IronMaiden>
*/
class Shaders {

    public:

        /**
            @brief This class is a Singleton Pattern class, everybody can access to it.
        */
        static Shaders & get_ref() {
            static Shaders m_shader;
            return m_shader;
        }

        ~Shaders();

    private:

        ///Container which stores shaders
        map<GLuint, Shader> m_container;

        ///Stores the last log produced by the compilation of the shaders.
        string m_log;

        ///Constructor.
        Shaders();

    public:

        /**
            @brief Given the filename(s) of the shader(s), this method creates a shader program.
            Note: Only a filename is needed, and is recommended that the name of the file be the
            same for each shader, the unique difference should be the extensions:
                        Vertex Shader    --> my_shader.vert
                        Fragment Shader --> my_shader.frag
                        Geometry Shader --> my_shader.geom
            @param vertex_filename [IN]: Filename of the code where is the vertex shader (Use "" for no vertex shader).
            @param fragment_filename [IN]: Filename of the code where is the fragment shader (Use "" for no fragment shader).
            @return The numeric identifier of the shader.
        */
        GLuint load_shader(string vertex_filename, string fragment_filename);

        GLuint load_shader_src(string vertex_text = "", string fragment_text = "");

        /**
            @brief Given a shader name (previously loaded successfully), enables this shader in the GPU.
            Also, disables the previous shader which was working in the GPU.
            @param name [IN]: The name of the shader.
            @return The numeric identifier of the shader.
        */
        void enable_shader(GLuint shader_number);

        /**
            @brief Disables shader in the GPU, setting the fixed pipeline (Default).
            Disables the previous shader which was working in the GPU.
        */
        void disable_shader();

        /**
            @brief Disables shader in the GPU, setting the fixed pipeline (Default).
            @param name [IN]: The name of the shader.
        */
        void unload_shader(GLuint & shader_number);

        /**
            @brief Gets the log of the shaders (Compilation problems, and so on.)
            @return The log (compilation errors and so on)
        */
        string log();

        /**
            @brief Empties the log.
        */
        void clear_log();
};

#endif
