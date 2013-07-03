
#include <stdio.h>
#include <stdlib.h>
#include <set>
#include <iostream>

#include <voxelDeformer.hpp>

using namespace std;

static int g_last_x = 0;
static int g_last_y = 0;
static float g_rotation_x = 0.0f;
static float g_rotation_y = 0.0f;
static bool g_mouseLeftDown = false;
static bool g_mouseRightDown = false;
static bool g_mouseMiddleDown = false;

static float g_camera_distance = -5.0;

static unsigned g_window_width = 700;
static unsigned g_window_height = 700;


static bool g_tetramesh_display = true;
static tetramesh g_tetramesh;
static vector<vec3> g_current_nodes;
static set<vec3 *> g_selected_vertices;
static vec3 * g_active_vertex = NULL;
static aabb g_tetramesh_aabb;

static voxelDeformer g_deformer;
static voxelVolume g_voxel_volume;
static iaabb g_voxel_volume_aabb;

static unsigned g_volume_draw_shader_id = 0;


//--------------- 3D Textures ---------------

GLuint g_tex_3d = 0;





//------------- End 3D Textures ---------------


void reshape(int w, int h) {
    // Prevent a divide by zero, when window is too short
    // (you cant make a window of zero width).
    if(h == 0)
        h = 1;
    float ratio = 1.0 * w / h;
    // Reset the coordinate system before modifying
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45, ratio, 1, 1000);
    glViewport(0, 0, w, h);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, g_camera_distance);
    glRotatef(g_rotation_x, 1, 0, 0);
    glRotatef(g_rotation_y, 0, 1, 0);
    g_window_width = w;
    g_window_height = h;
}


void draw() {

    //Limpia el buffer de profundidad poniendo el maximo valor de profundidad.
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, g_camera_distance);
    glRotatef(g_rotation_x, 1, 0, 0);
    glRotatef(g_rotation_y, 0, 1, 0);

    //----------------- Drawing the tetramesh -----------------

    if (g_tetramesh_display == true) {
        //Painting Tetramesh information.
        glColor4f(0.0, 1.0, 0.0, 0.5);
        draw_edges(g_tetramesh.e, g_current_nodes);

        glEnable(GL_POINT_SMOOTH);
        glColor4f(0.0, 1.0, 1.0, 1.0);
        glPointSize(18.0);
        draw_nodes(g_current_nodes);

        //Drawing the active node.
        if (g_active_vertex != NULL) {
            glColor4f(1.0, 0.0, 0.0, 1.0);
            glPointSize(30.0);
            glBegin(GL_POINTS);
                glVertex3f(g_active_vertex->x, g_active_vertex->y, g_active_vertex->z);
            glEnd();
        }

        glPointSize(1.0);
        glDisable(GL_POINT_SMOOTH);
    }

    //----------------- Drawing the volume -----------------
    draw_volume(g_tetramesh_aabb,
                g_tex_3d,
                g_volume_draw_shader_id);

    glutSwapBuffers();
}


void keyboard(unsigned char key, int x, int y) {
    switch (key){
        case ' ':
            {
                //Making the deformation.
                voxelVolume new_volume;
                //g_deformer.deform(g_current_nodes, new_volume);
                g_deformer.deform_tex3d(g_current_nodes, new_volume);
                //Putting the new volume as default display volume.
                unload_texture_3d(g_tex_3d);
                g_tex_3d = load_texture_3d(new_volume.width(),
                                           new_volume.height(),
                                           new_volume.depth(),
                                           new_volume.texels().data());
                new_volume.clear();

                //Updating the aabb of the tetramesh.
                g_tetramesh_aabb.set(g_current_nodes);
            }
            break;

        case 27:
            Shaders::get_ref().unload_shader(g_volume_draw_shader_id);
            exit(0);
            break;
    }
    glutPostRedisplay();
}


void special_keyboard(int key, int x, int y) {
    switch (key){
        case GLUT_KEY_F1:
            exit(0);
            break;
    }
    glutPostRedisplay();
}


void select_vertices(int x, int y) {

    //cout << "mouse: ("<< x << ", " << y << ")" << endl;
    //Con esto cogemos el plano de proyección, que nos indica los objetos que se ven.
    GLint viewport[4];
    glGetIntegerv (GL_VIEWPORT, viewport);

    //Indicamos a openGL que anote los toques en nuestro buffer.
    GLuint selectBuf[512];
    glSelectBuffer (512, selectBuf);

    //Le decimos a OpenGL que ahora vamos a pintar los objetos con intención
    //de averiguar los toques.
    glRenderMode (GL_SELECT);

    //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, g_camera_distance);
    glRotatef(g_rotation_x, 1, 0, 0);
    glRotatef(g_rotation_y, 0, 1, 0);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    //glViewport(0, 0, g_window_width, g_window_height);

    //Ahora vamos a crear un rayo que saldrá desde donde está el cursor al final de la escena.
    //el rayo tiene un grosor de 5x5 píxels, solamente se pintará la parte de la escena que
    //esté dentro de este volumen.
    //X e Y son las coordenadas del ratón en pantalla.
    gluPickMatrix (x, (viewport[3] - y), 40.0, 40.0, viewport);

    gluPerspective(45, 1.0 * g_window_width / g_window_height, 1, 1000);

    //Reseteamos el contador de números.
    glInitNames();
    vec3 p;

    glPointSize(18.0);
    for (unsigned i=0; i<g_current_nodes.size(); i++) {
        glPushName(i);
        glBegin(GL_POINTS);
            p = g_current_nodes[i];
            glVertex3f(p.x, p.y, p.z);
        glEnd();
        glPopName();
    }
    glPointSize(1.0);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glFlush();

    GLint hits = glRenderMode (GL_RENDER);
    GLint object_id = 0;
    //GLuint numero_partes_tocadas = 0;
    //float min_depth, max_depth;

    if (hits > 0) {
        //cout << "Touched "<< hits << " objects." << endl;
        GLuint toque = 0;
        for (int i = 0; i < hits; i++) {
            //numero_partes_tocadas = selectBuf[toque];
            toque++;
            //min_depth = (float) selectBuf[toque]/0x7fffffff;
            toque++;
            //max_depth = (float) selectBuf[toque]/0x7fffffff;
            toque++;

            object_id = selectBuf[toque];
            //printf ("  Object %d  depth [%f, %f]\n", object_id, min_depth, max_depth);

            set<vec3 *>::iterator it = g_selected_vertices.find(&g_current_nodes[object_id]);
            if (it == g_selected_vertices.end()) {
                //Seleccionando el objeto.
                g_selected_vertices.insert(&g_current_nodes[object_id]);
                g_active_vertex = &g_current_nodes[object_id];
            } else {
                //Deseleccionando el objeto.
                g_selected_vertices.erase(it);
                g_active_vertex = NULL;
            }
            toque++;
        }
        //cout << endl << endl;
    }
    //Repintamos la escena de modo normal y corriente.
    glutPostRedisplay();
}


void mouse_click(int button, int state, int x, int y) {
    g_last_x = x;
    g_last_y = y;

    if(button == GLUT_LEFT_BUTTON) {
        if(state == GLUT_DOWN) {
            g_mouseLeftDown = true;
        } else if(state == GLUT_UP)
            g_mouseLeftDown = false;
    } else {
        if(button == GLUT_RIGHT_BUTTON) {
            if(state == GLUT_DOWN) {
                g_mouseRightDown = true;
                //Iniciando el modo de seleccion de vertices.
                if (g_tetramesh_display == true) {
                    select_vertices(x, y);
                }
            } else {
                if(state == GLUT_UP) {
                    g_mouseRightDown = false;
                    g_active_vertex = NULL;
                }
            }
        } else {
            if(button == GLUT_MIDDLE_BUTTON) {
                if(state == GLUT_DOWN) {
                    g_mouseMiddleDown = true;
                } else {
                    if(state == GLUT_UP) {
                        g_mouseMiddleDown = false;
                    }
                }
            }
        }
    }
}



void grab_vertex(vec3 * p, int x, int y) {

    static GLfloat model_view_mat[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, model_view_mat);
    matrix mv_mat(4, 4, model_view_mat);
    mv_mat.transpose();

    static GLfloat proj_mat[16];
    glGetFloatv(GL_PROJECTION_MATRIX, proj_mat);
    matrix pj_mat(4, 4, proj_mat);
    pj_mat.transpose();

    matrix T;
    mul(T, pj_mat, mv_mat);
    vec4 v, q;

    v.set(*p);
    v.w = 1.0;

    mul(q, T, v);

    //Getting the depth of the vextex (camera coordinates).
    q.x = (x * 2.0 / g_window_width) - 1.0;
    q.y = (((int) g_window_height - y) * 2.0 / g_window_height) - 1.0;

    q.x *= q.w;
    q.y *= q.w;

    T.invert();
    mul(v, T, q);

    p->set(v.x, v.y, v.z);

    //Making the deformation.
    voxelVolume new_volume;
    //g_deformer.deform(g_current_nodes, new_volume);
    g_deformer.deform_tex3d(g_current_nodes, new_volume);


    //Putting the new volume as default display volume.
    unload_texture_3d(g_tex_3d);
    g_tex_3d = load_texture_3d(new_volume.width(),
                               new_volume.height(),
                               new_volume.depth(),
                               new_volume.texels().data());
    new_volume.clear();

    //Updating the aabb of the tetramesh.
    g_tetramesh_aabb.set(g_current_nodes);
}




void mouse_motion(int x, int y) {
    if (g_mouseLeftDown) {
        //Rotating the camera around our scene.
        g_rotation_x += (y - g_last_y) * 0.5;
        g_rotation_y += (x - g_last_x) * 0.5;

        if (g_rotation_x < 0.0f) {
            g_rotation_x += 360.0f;
        }
        if (g_rotation_x > 360.0f) {
            g_rotation_x -= 360.0f;
        }

        if (g_rotation_y < 0.0f) {
            g_rotation_y += 360.0f;
        }
        if (g_rotation_y > 360.0f) {
            g_rotation_y -= 360.0f;
        }
    }
    if (g_mouseMiddleDown) {
        g_camera_distance += (y - g_last_y) * 0.2f;
    }
    if (g_mouseRightDown) {
        //Arrastrando el vertice activo.
        if (g_active_vertex != NULL) {
            //Desplazando el vértice a donde se mueva el ratón.
            grab_vertex(g_active_vertex, x, y);
        }
    }
    g_last_x = x;
    g_last_y = y;

    glutPostRedisplay();
}



int main (int argc, char **argv) {

    cout << "------------- VoxelDeformer Simple Demo -------------" << endl << endl;
    cout << "  COMANDS/ACTIONS:" << endl;
    cout << "       Rotate the volume: move the mouse while the left button is pressed." << endl;
    cout << "       Zoom: move the mouse while the middle button is pressed." << endl;
    cout << "       Deform a node: click you mouse on a cyan node with right button and drag it." << endl << endl;

    omp_set_nested(1);
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);

    glutInitWindowPosition(200, 200);
    glutInitWindowSize(g_window_width, g_window_height);
    glutCreateWindow("Voxel Volume Viewer");
    glClearColor(0.0, 0.0, 0.0, 0.0);

    //BUGFIX:
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glutDisplayFunc(draw);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutSpecialFunc(special_keyboard);

    glutMouseFunc(mouse_click);
    //This function captures the movement of the mouse with almost a button pushed.
    glutMotionFunc(mouse_motion);

    //This function captures the movement of the mouse.
    glutPassiveMotionFunc(mouse_motion);

    //glShadeModel(GL_SMOOTH);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    if (!g_voxel_volume.load("Bucky_32x32x32x1.raw", 32, 32, 32)) {
        cerr << "ERROR: Filename could not load :-(" << endl;
        exit(-1);
        return -1;
    }

    g_tex_3d = load_texture_3d(g_voxel_volume.width(),
                               g_voxel_volume.height(),
                               g_voxel_volume.depth(),
                               g_voxel_volume.texels().data());

    g_tetramesh_aabb.aabb_min.set(-1.0, -1.0, -1.0);
    g_tetramesh_aabb.aabb_max.set(1.0, 1.0, 1.0);
    g_tetramesh.tetrahedralize(g_tetramesh_aabb, 1);
//TODO: Put the next line inside the tetramesh?
    g_tetramesh.update_edges();
    g_current_nodes.assign(g_tetramesh.x.begin(), g_tetramesh.x.end());

    g_deformer.init(&g_voxel_volume, &g_tetramesh);

    g_volume_draw_shader_id = Shaders::get_ref().load_shader("simpleDemo_ray_cast.vert",
                                                             "simpleDemo_ray_cast_3D.frag");

    glutMainLoop();
    return 0;
}
