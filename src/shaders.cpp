//
// C++ Implementation: shaders
//
// Description:
//
//
// Author: Jorge Gascon Perez,,, <jorge@IronMaiden>, (C) 2009
//
// Copyright: See COPYING file that comes with this distribution
//
//
#include <shaders.hpp>

Shaders::Shaders() {
    this->m_container.clear();
    this->m_log = "";
}


Shaders::~Shaders() {
    map<GLuint, Shader>::iterator it;
    for (it = this->m_container.begin(); it != this->m_container.end(); it++) {
        unload_shader((GLuint&) it->first);
    }
}


GLuint Shaders::load_shader(string vertex_filename,
                            string fragment_filename) {

    cout << "    load_shaders: " << vertex_filename << " and " << fragment_filename << endl;
    GLsizei LogLength = 500;
    GLchar compilationLog[LogLength];
    GLsizei lengthObtained = 0;

    FILE * f;
    unsigned int count=0;
    char * content = NULL;
    Shader new_shader;
    new_shader.id = glCreateProgram();

    //Creating the vertex shader
    if (vertex_filename != "") {
        f = fopen(vertex_filename.c_str(), "rt");
        //If loaded successfully, looking for if this shader is loaded yet.
        if (f == NULL) {
            cout << "ERROR: Shaders::load_shader --> File not found " << vertex_filename << endl;
        } else {
            fseek(f, 0, SEEK_END);
            count = ftell(f);
            rewind(f);
            if (count > 0) {
                content = (char *)malloc(sizeof(char) * (count+1));
                count = fread(content, sizeof(char), count, f);
                content[count] = '\0';

                //Creating the shader.
                new_shader.vertex_id = glCreateShader(GL_VERTEX_SHADER);
                const char * code = content;
                glShaderSourceARB(new_shader.vertex_id, 1, &code, NULL);
                free(content);
                content = NULL;

                //Compiling the shader.
                glCompileShaderARB(new_shader.vertex_id);
                glGetShaderInfoLog(new_shader.vertex_id, LogLength, &lengthObtained, compilationLog);
                if (lengthObtained>0) { printf("Log del Vertex Shader \n %s\n", compilationLog); }
                this->m_log += compilationLog;
                glAttachShader(new_shader.id, new_shader.vertex_id);
            }
        }
        fclose(f);
    }


    //Creating the fragment shader
    if (fragment_filename != "") {
        f = fopen(fragment_filename.c_str(), "rt");
        //If loaded successfully, looking for if this shader is loaded yet.
        if (f == NULL) {
            cout << "ERROR: Shaders::load_shader --> File not found " << fragment_filename << endl;
        } else {

            fseek(f, 0, SEEK_END);
            count = ftell(f);
            rewind(f);
            if (count > 0) {
                content = (char *)malloc(sizeof(char) * (count+1));
                count = fread(content, sizeof(char), count, f);
                content[count] = '\0';

                //Creating the shader.
                new_shader.fragment_id = glCreateShader(GL_FRAGMENT_SHADER);
                const char * code = content;
                glShaderSourceARB(new_shader.fragment_id, 1, &code, NULL);
                free(content);
                content = NULL;

                //Compiling the shader.
                glCompileShaderARB(new_shader.fragment_id);
                glGetShaderInfoLog(new_shader.fragment_id, LogLength, &lengthObtained, compilationLog);
                if (lengthObtained>0) { printf("Log del Fragment Shader \n %s\n", compilationLog); }
                this->m_log += compilationLog;
                glAttachShader(new_shader.id, new_shader.fragment_id);
            }
        }
        fclose(f);
    }
    //Se enlaza el programa y ya estamos listos.
    glLinkProgramARB(new_shader.id);
    //glUseProgram(new_shader.id);
    this->m_container[new_shader.id] = new_shader;
    return new_shader.id;
}





GLuint Shaders::load_shader_src(string vertex_text, string fragment_text) {
    cout << "Shaders::load_shader_src" << endl;
    GLsizei LogLength = 500;
    GLchar compilationLog[LogLength];
    GLsizei lengthObtained = 0;

    Shader new_shader;
    new_shader.id = glCreateProgram();

    //Creating the vertex shader
    if (vertex_text != "") {
        //Creating the shader.
        new_shader.vertex_id = glCreateShader(GL_VERTEX_SHADER);
        const char * code = vertex_text.c_str();
        glShaderSourceARB(new_shader.vertex_id, 1, &code, NULL);

        //Compiling the shader.
        glCompileShaderARB(new_shader.vertex_id);
        glGetShaderInfoLog(new_shader.vertex_id, LogLength, &lengthObtained, compilationLog);
        if (lengthObtained>0) { printf("Log del Vertex Shader \n %s\n", compilationLog); }
        this->m_log += compilationLog;
        glAttachShader(new_shader.id, new_shader.vertex_id);
    }


    //Creating the fragment shader
    if (fragment_text != "") {
        //Creating the shader.
        new_shader.fragment_id = glCreateShader(GL_FRAGMENT_SHADER);
        const char * code = fragment_text.c_str();
        glShaderSourceARB(new_shader.fragment_id, 1, &code, NULL);

        //Compiling the shader.
        glCompileShaderARB(new_shader.fragment_id);
        glGetShaderInfoLog(new_shader.fragment_id, LogLength, &lengthObtained, compilationLog);
        if (lengthObtained>0) { printf("Log del Fragment Shader \n %s\n", compilationLog); }
        this->m_log += compilationLog;
        glAttachShader(new_shader.id, new_shader.fragment_id);
    }

    //Se enlaza el programa y ya estamos listos.
    glLinkProgramARB(new_shader.id);
    this->m_container[new_shader.id] = new_shader;
    return new_shader.id;
}





void Shaders::enable_shader(GLuint shader_number) {
    if (this->m_container.find(shader_number) != this->m_container.end()) {
        glUseProgram(this->m_container[shader_number].id);
    } else {
        cout << "Shaders::enable_shader --> ERROR: Shader with number " << shader_number << " does not exist.\n";
    }
}


void Shaders::disable_shader() {
    //Setting the default fixed GPU pipeline.
    glUseProgram(0);
}


void Shaders::unload_shader(GLuint & shader_number) {
    this->disable_shader();
    if (this->m_container.find(shader_number) != this->m_container.end()) {
        Shader old_shader = this->m_container[shader_number];
        if (old_shader.vertex_id != 0) {
            glDetachShader(old_shader.id, old_shader.vertex_id);
            glDeleteShader(old_shader.vertex_id);
        }

        if (old_shader.fragment_id != 0) {
            glDetachShader(old_shader.id, old_shader.fragment_id);
            glDeleteShader(old_shader.fragment_id);
        }
        glDeleteProgram(old_shader.id);
        shader_number = 0;
    }
}


string Shaders::log() {
    return this->m_log;
}


void Shaders::clear_log() {
    this->m_log = "";
}
