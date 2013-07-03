varying vec3 pos;

void main() {
    gl_Position = ftransform();
    pos = vec3(gl_Vertex);
}
