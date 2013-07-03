
#include <types.hpp>
#include <matrix.hpp>
#include <cstring>


string replace (string entrada, string busca, string cambia) {
    if (entrada != "" && busca != cambia) {
        string::size_type pos=0;
        pos = entrada.find(busca,pos);
        while (pos != string::npos) {
            entrada.replace(pos,busca.length(),cambia);
            pos = entrada.find(busca,pos+cambia.length());
        }
    }
    return entrada;
}


string int_to_str (const int v) {
    static char s_v[100];
    sprintf( s_v, "%d", v);
    return replace (s_v, " ", "");
}


int str_to_int (string v) {
    if (v == "" || v == " ") {
        return 0;
    }
    return atoi(v.c_str());
}


string double_to_str (const double v) {
    static char s_v[100];
    sprintf( s_v, "%lf", v);
    return replace (s_v, " ", "");
}



double str_to_double (string v) {
    if (v == "" || v == " ") {
        return 0.0f;
    }
    return atof(v.c_str());
}



string float_to_str (const float v) {
    static char s_v[100];
    sprintf( s_v, "%f", v);
    return replace (s_v, " ", "");
}


string to_binary_format (const unsigned char v) {
    string r = "";
    unsigned char vc = v;
    unsigned char m;
    for (unsigned i=0; i<8; i++) {
        m = vc % 2;
        vc = vc / 2;
        r = ((m != 0) ? "1" : "0") + r;
    }
    return r;
}


vec2::vec2(double _x, double _y) {
    this->set(_x, _y);
}



double vec2::length() {
    return sqrt(this->x * this->x +
                this->y * this->y);
}


double vec2::length2() {
    return (this->x * this->x +
            this->y * this->y);
}


string vec2::to_s() {
    string result = "(";
    result += double_to_str(this->x);
    result += ", ";
    result += double_to_str(this->y);
    result += ")";
    return result;
}



void vec2::set(double _x, double _y) {
    this->x = _x;
    this->y = _y;
}


void vec2::set(const vec2 & o) {
    this->x = o.x;
    this->y = o.y;
}



void vec2::normalize() {
    double length = sqrt(this->x * this->x +
                         this->y * this->y);
    if (length == 0.0) {
        return;
    }
    length = 1.0 / length;
    this->x *= length;
    this->y *= length;
}


double dot(vec2 * op1, vec2 * op2) {
    return (op1->x * op2->x + op1->y * op2->y);
}


void mul(vec2 * r, double v, const vec2 * op) {
    r->x = v * op->x;
    r->y = v * op->y;
}


void mul(vec2 & r, double v, const vec2 & op) {
    r.x = v * op.x;
    r.y = v * op.y;
}



void add(vec2 * r, const vec2 * op1, const vec2 * op2) {
    r->x = op1->x + op2->x;
    r->y = op1->y + op2->y;
}


void add(vec2 & r, const vec2 & op1, const vec2 & op2) {
    r.x = op1.x + op2.x;
    r.y = op1.y + op2.y;
}


void sub(vec2 * r, const vec2 * op1, const vec2 * op2) {
    r->x = op1->x - op2->x;
    r->y = op1->y - op2->y;
}


void sub(vec2 & r, const vec2 & op1, const vec2 & op2) {
    r.x = op1.x - op2.x;
    r.y = op1.y - op2.y;
}




string vec3::to_s() const {
    string result = "(";
    result += double_to_str(this->x);
    result += ", ";
    result += double_to_str(this->y);
    result += ", ";
    result += double_to_str(this->z);
    result += ")";
    return result;
}





vec3 vec3::operator * (const vec3 other) {
    static vec3 r;
    r.x = this->x * other.x;
    r.y = this->y * other.y;
    r.z = this->z * other.z;
    return r;
}


double dot(vec3 * op1, vec3 * op2) {
    return (op1->x * op2->x + op1->y * op2->y + op1->z * op2->z);
}


void mul(vec3 * r, double v, const vec3 * op) {
    r->x = v * op->x;
    r->y = v * op->y;
    r->z = v * op->z;
}


void mul(vec3 & r, double v, const vec3 & op) {
    r.x = v * op.x;
    r.y = v * op.y;
    r.z = v * op.z;
}


void cross(vec3 * r, const vec3 * op1, const vec3 * op2) {
    r->x = op1->y * op2->z - op1->z * op2->y;
    r->y = op1->z * op2->x - op1->x * op2->z;
    r->z = op1->x * op2->y - op1->y * op2->x;
}


void add(vec3 * r, const vec3 * op1, const vec3 * op2) {
    r->x = op1->x + op2->x;
    r->y = op1->y + op2->y;
    r->z = op1->z + op2->z;
}


void add(vec3 & r, const vec3 & op1, const vec3 & op2) {
    r.x = op1.x + op2.x;
    r.y = op1.y + op2.y;
    r.z = op1.z + op2.z;
}




void add(ivec3 & r, const ivec3 & op1, const ivec3 & op2) {
    r.x = op1.x + op2.x;
    r.y = op1.y + op2.y;
    r.z = op1.z + op2.z;
}



void sub(vec3 * r, const vec3 * op1, const vec3 * op2) {
    r->x = op1->x - op2->x;
    r->y = op1->y - op2->y;
    r->z = op1->z - op2->z;
}


void sub(vec3 & r, const vec3 & op1, const vec3 & op2) {
    r.x = op1.x - op2.x;
    r.y = op1.y - op2.y;
    r.z = op1.z - op2.z;
}



void sub(ivec3 & r, const ivec3 & op1, const ivec3 & op2) {
    r.x = op1.x - op2.x;
    r.y = op1.y - op2.y;
    r.z = op1.z - op2.z;
}






uvec3::uvec3(unsigned _x, unsigned _y, unsigned _z) {
    this->set(_x, _y, _z);
}


void uvec3::set(unsigned _x, unsigned _y, unsigned _z) {
    this->x = _x;
    this->y = _y;
    this->z = _z;
}




string uvec3::to_s() {
    string result = "(";
    result += int_to_str(this->x);
    result += ", ";
    result += int_to_str(this->y);
    result += ", ";
    result += int_to_str(this->z);
    result += ")";
    return result;
}






uvec2::uvec2(unsigned _x, unsigned _y) {
    this->set(_x, _y);
}


void uvec2::set(unsigned _x, unsigned _y) {
    this->x = _x;
    this->y = _y;
}


unsigned uvec2::length2() {
    return (this->x * this->x + this->y * this->y);
}


void uvec2::set(const uvec2 & o) {
    this->x = o.x;
    this->y = o.y;
}




ivec3::ivec3(int _x, int _y, int _z) {
    this->set(_x, _y, _z);
}


void ivec3::set(int _x, int _y, int _z) {
    this->x = _x;
    this->y = _y;
    this->z = _z;
}


int ivec3::length2() {
    return (this->x * this->x + this->y * this->y + this->z * this->z);
}


void ivec3::set(const ivec3 & o) {
    this->x = o.x;
    this->y = o.y;
    this->z = o.z;
}


string ivec3::to_s() {
    string result = "(";
    result += int_to_str(this->x);
    result += ", ";
    result += int_to_str(this->y);
    result += ", ";
    result += int_to_str(this->z);
    result += ")";
    return result;
}



vec4::vec4(double _x, double _y, double _z, double _w) {
    this->set(_x, _y, _z, _w);
}


void vec4::set(double _x, double _y, double _z, double _w) {
    this->x = _x;
    this->y = _y;
    this->z = _z;
    this->w = _w;
}


void vec4::set(const vec4 & v) {
    this->x = v.x;
    this->y = v.y;
    this->z = v.z;
    this->w = v.w;
}


void vec4::set(const vec3 & v) {
    this->x = v.x;
    this->y = v.y;
    this->z = v.z;
    this->w = 1.0;
}


string vec4::to_s() {
    string result = "(";
    result += double_to_str(this->x);
    result += ", ";
    result += double_to_str(this->y);
    result += ", ";
    result += double_to_str(this->z);
    result += ", ";
    result += double_to_str(this->w);
    result += ")";
    return result;
}


ivec4::ivec4(int _x, int _y, int _z, int _w) {
    this->set(_x, _y, _z, _w);
}



void ivec4::set(int _x, int _y, int _z, int _w) {
    this->x = _x;
    this->y = _y;
    this->z = _z;
    this->w = _w;
}


string ivec4::to_s() {
    string result = "(";
    result += int_to_str(this->x);
    result += ", ";
    result += int_to_str(this->y);
    result += ", ";
    result += int_to_str(this->z);
    result += ", ";
    result += int_to_str(this->w);
    result += ")";
    return result;
}



uvec4::uvec4(unsigned _x, unsigned _y, unsigned _z, unsigned _w) {
    this->set(_x, _y, _z, _w);
}


void uvec4::set(unsigned _x, unsigned _y, unsigned _z, unsigned _w) {
    this->x = _x;
    this->y = _y;
    this->z = _z;
    this->w = _w;
}


string uvec4::to_s() {
    string result = "(";
    result += int_to_str(this->x);
    result += ", ";
    result += int_to_str(this->y);
    result += ", ";
    result += int_to_str(this->z);
    result += ", ";
    result += int_to_str(this->w);
    result += ")";
    return result;
}


aabb::aabb(double min_x, double min_y, double min_z,
           double max_x, double max_y, double max_z) {
    this->aabb_min.set(min_x, min_y, min_z);
    this->aabb_max.set(max_x, max_y, max_z);
}


aabb::aabb(const vector<vec3> & v) {
    this->set(v);
}


aabb::~aabb() {}


bool aabb::is_outside(aabb & other) {
    return (this->aabb_min.x < other.aabb_min.x ||
            this->aabb_min.y < other.aabb_min.y ||
            this->aabb_min.z < other.aabb_min.z ||
            this->aabb_max.x > other.aabb_max.x ||
            this->aabb_max.y > other.aabb_max.y ||
            this->aabb_max.z > other.aabb_max.z);
}



void aabb::adapt(double new_size_x, double new_size_y, double new_size_z) {
    double new_size = (new_size_x - (aabb_max.x - aabb_min.x)) / 2;
    aabb_min.x -= new_size;
    aabb_max.x += new_size;

    new_size = (new_size_y - (aabb_max.y - aabb_min.y)) / 2;
    aabb_min.y -= new_size;
    aabb_max.y += new_size;

    new_size = (new_size_z - (aabb_max.z - aabb_min.z)) / 2;
    aabb_min.z -= new_size;
    aabb_max.z += new_size;
}


void aabb::clear() {
    this->aabb_min.set(0.0, 0.0, 0.0);
    this->aabb_max.set(0.0, 0.0, 0.0);
}



vec3 aabb::midpoint() {
    vec3 result;
    result.x = (this->aabb_max.x - this->aabb_min.x) * 0.5;
    result.y = (this->aabb_max.y - this->aabb_min.y) * 0.5;
    result.z = (this->aabb_max.z - this->aabb_min.z) * 0.5;
    return result;
}


vec3 aabb::get_node_th(unsigned i) {
    vec3 r;
    switch(i) {
        case 0:
            r = this->aabb_min;
            break;

        case 1:
            r.x = this->aabb_min.x;
            r.y = this->aabb_min.y;
            r.z = this->aabb_max.z;
            break;

        case 2:
            r.x = this->aabb_min.x;
            r.y = this->aabb_max.y;
            r.z = this->aabb_min.z;
            break;

        case 3:
            r.x = this->aabb_min.x;
            r.y = this->aabb_max.y;
            r.z = this->aabb_max.z;
            break;

        case 4:
            r.x = this->aabb_max.x;
            r.y = this->aabb_min.y;
            r.z = this->aabb_min.z;
            break;

        case 5:
            r.x = this->aabb_max.x;
            r.y = this->aabb_min.y;
            r.z = this->aabb_max.z;
            break;

        case 6:
            r.x = this->aabb_max.x;
            r.y = this->aabb_max.y;
            r.z = this->aabb_min.z;
            break;

        default:
            r = this->aabb_max;
            break;
    }
    return r;
}


void aabb::set(const double x, const double y, const double z) {
    aabb_min.set(x,y,z);
    aabb_max.set(x,y,z);
}



void aabb::set(const vec3 & v) {
    aabb_min.set(v);
    aabb_max.set(v);
}


void aabb::add(const vec3 & v) {
    this->aabb_min.x = (v.x < this->aabb_min.x) ? v.x : this->aabb_min.x;
    this->aabb_min.y = (v.y < this->aabb_min.y) ? v.y : this->aabb_min.y;
    this->aabb_min.z = (v.z < this->aabb_min.z) ? v.z : this->aabb_min.z;

    this->aabb_max.x = (v.x > this->aabb_max.x) ? v.x : this->aabb_max.x;
    this->aabb_max.y = (v.y > this->aabb_max.y) ? v.y : this->aabb_max.y;
    this->aabb_max.z = (v.z > this->aabb_max.z) ? v.z : this->aabb_max.z;
}



void aabb::set(const vector<vec3> & v) {
    if (v.size() > 0) {
        this->aabb_min.set(v[0]);
        this->aabb_max.set(v[0]);
        this->add(v);
    }
}


void aabb::add(const vector<vec3> & v) {
    for (unsigned i=0; i<v.size(); i++) {
        this->add(v[i]);
    }
}

void aabb::set(const vec3 v[4]) {
    this->aabb_min.set(v[0]);
    this->aabb_max.set(v[0]);
    for (unsigned i=1; i<4 ; i++)
        this->add(v[i]);
}


string aabb::to_s() {
    string result = "AABB:";
    result += " min: " + this->aabb_min.to_s();
    result += " max: " + this->aabb_max.to_s();
    return result;
}


double aabb::width() const {
    return this->aabb_max.x - this->aabb_min.x;
}

double aabb::height() const {
    return this->aabb_max.y - this->aabb_min.y;
}

double aabb::depth() const {
    return this->aabb_max.z - this->aabb_min.z;
}


unsigned aabb::iwidth() const {
    return CEIL_POS(this->aabb_max.x - this->aabb_min.x);
}

unsigned aabb::iheight() const {
    return CEIL_POS(this->aabb_max.y - this->aabb_min.y);
}

unsigned aabb::idepth() const {
    return CEIL_POS(this->aabb_max.z - this->aabb_min.z);
}


vec3 aabb::dim() const {
    vec3 result(this->width(), this->height(), this->depth());
    return result;
}


uvec3 aabb::udim() const {
    uvec3 result(this->iwidth(), this->iheight(), this->idepth());
    return result;
}


unsigned aabb::size() const {
    return this->iwidth() * this->iheight() * this->idepth();
}


iaabb::iaabb(int min_x, int min_y, int min_z,
             int max_x, int max_y, int max_z) {
    this->set(min_x, min_y, min_z, max_x, max_y, max_z);
}


iaabb::~iaabb() {}


void iaabb::adapt(int new_size_x, int new_size_y, int new_size_z) {
    int new_size = (new_size_x - (aabb_max.x - aabb_min.x)) / 2;
    aabb_min.x -= new_size;
    aabb_max.x += new_size;

    new_size = (new_size_y - (aabb_max.y - aabb_min.y)) / 2;
    aabb_min.y -= new_size;
    aabb_max.y += new_size;

    new_size = (new_size_z - (aabb_max.z - aabb_min.z)) / 2;
    aabb_min.z -= new_size;
    aabb_max.z += new_size;
}


void iaabb::clear() {
    this->set();
}


void iaabb::set(int min_x, int min_y, int min_z,
                int max_x, int max_y, int max_z) {
    aabb_min.set(min_x, min_y, min_z);
    aabb_max.set(max_x, max_y, max_z);
}


void iaabb::set(const ivec3 & v) {
    aabb_min.set(v);
    aabb_max.set(v);
}


void iaabb::set(const vector<ivec3> & v) {
    if (v.size() > 0) {
        this->aabb_min.set(v[0]);
        this->aabb_max.set(v[0]);
        this->add(v);
    }
}


void iaabb::add(const vector<ivec3> & v) {
    for (unsigned i=0; i<v.size(); i++) {
        this->add(v[i]);
    }
}



void iaabb::add(const ivec3 & v) {
    this->aabb_min.x = (v.x < this->aabb_min.x) ? v.x : this->aabb_min.x;
    this->aabb_min.y = (v.y < this->aabb_min.y) ? v.y : this->aabb_min.y;
    this->aabb_min.z = (v.z < this->aabb_min.z) ? v.z : this->aabb_min.z;

    this->aabb_max.x = (v.x > this->aabb_max.x) ? v.x : this->aabb_max.x;
    this->aabb_max.y = (v.y > this->aabb_max.y) ? v.y : this->aabb_max.y;
    this->aabb_max.z = (v.z > this->aabb_max.z) ? v.z : this->aabb_max.z;
}



string iaabb::to_s() {
    string result = "AABB:";
    result += " min: " + this->aabb_min.to_s();
    result += " max: " + this->aabb_max.to_s();
    return result;
}


int iaabb::width() {
    return (this->aabb_max.x - this->aabb_min.x);
}

int iaabb::height() {
    return (this->aabb_max.y - this->aabb_min.y);
}

int iaabb::depth() {
    return (this->aabb_max.z - this->aabb_min.z);
}

uvec3 iaabb::dim() {
    uvec3 result(this->aabb_max.x - this->aabb_min.x,
                 this->aabb_max.y - this->aabb_min.y,
                 this->aabb_max.z - this->aabb_min.z);
    return result;
}




trimesh::trimesh() {
    this->clear();
}


trimesh::~trimesh() {
    this->clear();
}



aabb trimesh::get_aabb() {
    aabb result;
    result.set(this->x);
    return result;
}


void trimesh::clear() {
    this->x.clear();
    this->f.clear();
    this->n.clear();
    this->uv.clear();
    this->tex_ids.clear();
}



void trimesh::load(string filename) {

    //LIMITATION: For now, only .obj files loaded :-(

    FILE * file = fopen(filename.c_str(), "r");
    vector<vec3> vertices;
    vector<vec3> compact_v_normals;
    vector<vec3> face_normals;
    vector<vec3> vertices_normals;
    vector<vec2> uvs;
    vector<uvec3> uv_indexes;
    vector<uvec3> faces;
    char line[1000];
    int bytes_read;
    char * text_read;
    vec3 v;
    vec2 vt;
    float modulus;
    uvec3 f;
    uvec3 uv;
    uvec3 n;

    if (!file) {
        cout << "ERROR: load_obj: File [" << filename << "] not found.\n";
        return;
    }

    this->clear();

    //cout << "load_obj [" << filename << "]\n";
    while (!feof(file)) {
        bytes_read = fscanf(file, "%s", line);
        if (strcmp(line, "vn") == 0) {
            bytes_read = fscanf(file, "%lf %lf %lf\n", &v.x, &v.y, &v.z);
            compact_v_normals.push_back(v);
        } else if (strcmp(line, "vt") == 0) {
            bytes_read = fscanf(file, "%lf %lf\n", &vt.x, &vt.y);
            vt.y = 1.0f - vt.y;
            uvs.push_back(vt);
        } else if (strcmp(line, "v") == 0) {
            bytes_read = fscanf(file, "%lf %lf %lf\n", &v.x, &v.y, &v.z);
            //v.y = -v.y;
            vertices.push_back(v);
        } else if (strcmp(line, "f") == 0) {
            //f 7407//3970 7408//3979 7404//3978
            if (uvs.size() == 0) {
                //Reading a mesh without texture coordinates.
                bytes_read = fscanf(file, "%d//%d %d//%d %d//%d\n",
                                    &f.x, &n.x, &f.y, &n.y, &f.z, &n.z);
            } else {
                //f 13/14/1 14/12/1 16/2/1
                //Reading a mesh with texture coordinates.
                bytes_read = fscanf(file, "%d/%d/%d %d/%d/%d %d/%d/%d\n",
                                    &f.x, &uv.x, &n.x, &f.y, &uv.y, &n.y, &f.z, &uv.z, &n.z);
                uv.x--;
                uv.y--;
                uv.z--;
                uv_indexes.push_back(uv);
            }

            f.x--;
            f.y--;
            f.z--;

            faces.push_back(f);
            n.x--;
            n.y--;
            n.z--;

            v.x = compact_v_normals[n.x].x + compact_v_normals[n.y].x + compact_v_normals[n.z].x;
            v.y = compact_v_normals[n.x].y + compact_v_normals[n.y].y + compact_v_normals[n.z].y;
            v.z = compact_v_normals[n.x].z + compact_v_normals[n.y].z + compact_v_normals[n.z].z;
            modulus = 1.0 / sqrt(v.x * v.x + v.y * v.y + v.z * v.z);

            v.x *= modulus;
            v.y *= modulus;
            v.z *= modulus;

            face_normals.push_back(v);

        } else {
            text_read = fgets(line, 499, file);
        }
    }
    fclose(file);

    //If the object uses no texture, returning a non-textured mesh.
    if (uvs.size() == 0) {
        cout << "Read " << vertices.size() << " vertices.\n";
        cout << "Read " << face_normals.size() << " face normals.\n";
        cout << "Read " << faces.size() << " Faces.\n";
        this->x.assign(vertices.begin(), vertices.end());
        this->n.assign(face_normals.begin(), face_normals.end());
        this->f.assign(faces.begin(), faces.end());
    }

    //Duplicating vertices that have many texture coordinates.
    if (uv_indexes.size() != faces.size()) {
        cerr << "ERROR: load_obj() --> uv_indexes.size() != faces.size().\n";
    }

    //Creating a map which stores face-uv coincidences.
    string key;
    unsigned key_val;
    map<string, unsigned> coincidences;

    //Final list of vertices and faces.
    vector<vec3> final_vertices;
    vector<uvec3> final_faces;
    vector<vec2> final_uvs;

    final_faces.resize(faces.size());
    final_vertices.reserve(faces.size()*3);
    final_uvs.reserve(final_vertices.size());

    //Reading all the faces.
    for (unsigned i=0; i<faces.size(); i++) {
        for (unsigned j=0; j<3; j++) {
            key = int_to_str(faces[i][j]) + "_" + int_to_str(uv_indexes[i][j]);
            //cout << "Key: " << key << endl;
            //Searching the key of this vertex
            if (coincidences.find(key) == coincidences.end()) {
                //Key not found, creating new vertex and key.
                key_val = final_vertices.size();
                coincidences[key] = key_val;
                final_vertices.push_back(vertices[faces[i][j]]);
                final_uvs.push_back(uvs[uv_indexes[i][j]]);
            } else {
                //Key found, using found vertex.
                key_val = coincidences[key];
            }
            //Reasigning the position of the vertex of this face.
            final_faces[i][j] = key_val;
        }
    }

    cout << "load_obj: initial vertices: " << vertices.size() << endl;
    vertices.clear();
    vertices.assign(final_vertices.begin(), final_vertices.end());
    faces.clear();
    faces.assign(final_faces.begin(), final_faces.end());

    cout << "          unwrapped vertices: " << final_vertices.size() << endl;
    final_vertices.clear();
    final_faces.clear();
    coincidences.clear();

    cout << "Read " << vertices.size() << " vertices.\n";
    cout << "Read " << face_normals.size() << " face normals.\n";
    cout << "Read " << faces.size() << " Faces.\n";
    cout << "Read " << final_uvs.size() << " uvs.\n";

    cout << "Assigning vertices "<< vertices.size() <<"\n";
    this->x.assign(vertices.begin(), vertices.end());
    cout << "Assigning face normals "<< face_normals.size() <<"\n";
    this->n.assign(face_normals.begin(), face_normals.end());
    cout << "Assigning faces "<< faces.size() <<"\n";
    this->f.assign(faces.begin(), faces.end());
    cout << "Assigning uvs "<< final_uvs.size() <<"\n";
    this->uv.assign(final_uvs.begin(), final_uvs.end());

    vertices.clear();
    face_normals.clear();
    faces.clear();
    final_uvs.clear();
    compact_v_normals.clear();
    vertices_normals.clear();
    uvs.clear();
    uv_indexes.clear();
}



tetramesh::tetramesh() {
    this->clear();
}


tetramesh::~tetramesh() {
    this->clear();
}


void tetramesh::tetrahedralize(const aabb & bounds, unsigned divX, unsigned divY, unsigned divZ) {

    divY = (divY == 0) ? divX : divY;
    divZ = (divZ == 0) ? divX : divZ;

    //Building tetrahedra, each cell voxel has 5 tetrahedra.
    //Building vertices,
    this->x.resize((divX + 1) * (divY + 1) * (divZ + 1));
    this->t.resize(divX * divY * divZ * 5);

    unsigned index = 0;
    string label = "";
    map<string, unsigned> labels;

    //Compute the position of vertex
    for (unsigned k=0; k <= divZ; k++) {
        for (unsigned j=0; j <= divY; j++) {
            for (unsigned i=0; i <= divX; i++) {
                //Compute interpolation factor
                double fx = double (i) / double(divX);
                double fy = double (j) / double(divY);
                double fz = double (k) / double(divZ);

                //Interpolate the position of the vertices.
                this->x[index].x = bounds.aabb_min.x * (1-fx) + bounds.aabb_max.x * fx;
                this->x[index].y = bounds.aabb_min.y * (1-fy) + bounds.aabb_max.y * fy;
                this->x[index].z = bounds.aabb_min.z * (1-fz) + bounds.aabb_max.z * fz;

                //Registering the vertex position in order to be linked with tetrahedra.
                label = int_to_str(i) + "_" + int_to_str(j) + "_" + int_to_str(k);
                labels[label] = index;
                index++;
            }
        }
    }

    //Asigning the tetrahedra to their indices.
    index = 0;

    //Putting 5 tetras a voxel.
    for (unsigned k=0; k < divZ; k++) {
        for (unsigned j=0; j < divY; j++) {
            for (unsigned i=0; i < divX; i++) {

                if ((i+j+k) % 2 == 0) {

                    //Configuration 1 of the tetrahedra.

                    //First tetrahedron (central) (1,1,1) (1,0,0) (0,0,1) (0,1,0)
                    label = int_to_str(i+1) + "_" + int_to_str(j+1) + "_" + int_to_str(k+1);
                    this->t[index].x = labels[label];

                    label = int_to_str(i+1) + "_" + int_to_str(j) + "_" + int_to_str(k);
                    this->t[index].y = labels[label];

                    label = int_to_str(i) + "_" + int_to_str(j) + "_" + int_to_str(k+1);
                    this->t[index].z = labels[label];

                    label = int_to_str(i) + "_" + int_to_str(j+1) + "_" + int_to_str(k);
                    this->t[index].w = labels[label];

                    index++;


                    //Second tetrahedron (0,0,1) (1,0,1) (1,0,0) (1,1,1)
                    label = int_to_str(i) + "_" + int_to_str(j) + "_" + int_to_str(k+1);
                    this->t[index].x = labels[label];

                    label = int_to_str(i+1) + "_" + int_to_str(j) + "_" + int_to_str(k+1);
                    this->t[index].y = labels[label];

                    label = int_to_str(i+1) + "_" + int_to_str(j) + "_" + int_to_str(k);
                    this->t[index].z = labels[label];

                    label = int_to_str(i+1) + "_" + int_to_str(j+1) + "_" + int_to_str(k+1);
                    this->t[index].w = labels[label];

                    index++;


                    //Third tetrahedron (0,0,1) (1,1,1) (0,1,0) (0,1,1)
                    label = int_to_str(i) + "_" + int_to_str(j) + "_" + int_to_str(k+1);
                    this->t[index].x = labels[label];

                    label = int_to_str(i+1) + "_" + int_to_str(j+1) + "_" + int_to_str(k+1);
                    this->t[index].y = labels[label];

                    label = int_to_str(i) + "_" + int_to_str(j+1) + "_" + int_to_str(k);
                    this->t[index].z = labels[label];

                    label = int_to_str(i) + "_" + int_to_str(j+1) + "_" + int_to_str(k+1);
                    this->t[index].w = labels[label];

                    index++;


                    //Fourth tetrahedron (1,0,0) (0,0,0) (0,0,1) (0,1,0)
                    label = int_to_str(i+1) + "_" + int_to_str(j) + "_" + int_to_str(k);
                    this->t[index].x = labels[label];

                    label = int_to_str(i) + "_" + int_to_str(j) + "_" + int_to_str(k);
                    this->t[index].y = labels[label];

                    label = int_to_str(i) + "_" + int_to_str(j) + "_" + int_to_str(k+1);
                    this->t[index].z = labels[label];

                    label = int_to_str(i) + "_" + int_to_str(j+1) + "_" + int_to_str(k);
                    this->t[index].w = labels[label];

                    index++;


                    //Fifth tetrahedron (1,0,0) (0,1,0) (1,1,1) (1,1,0)
                    label = int_to_str(i+1) + "_" + int_to_str(j) + "_" + int_to_str(k);
                    this->t[index].x = labels[label];

                    label = int_to_str(i) + "_" + int_to_str(j+1) + "_" + int_to_str(k);
                    this->t[index].y = labels[label];

                    label = int_to_str(i+1) + "_" + int_to_str(j+1) + "_" + int_to_str(k+1);
                    this->t[index].z = labels[label];

                    label = int_to_str(i+1) + "_" + int_to_str(j+1) + "_" + int_to_str(k);
                    this->t[index].w = labels[label];

                    index++;


                } else {

                    //Configuration 2 of the tetrahedra.
                    //First tetrahedron (central) (0,0,0) (0,1,1) (1,0,1) (1,1,0)
                    label = int_to_str(i) + "_" + int_to_str(j) + "_" + int_to_str(k);
                    this->t[index].x = labels[label];

                    label = int_to_str(i) + "_" + int_to_str(j+1) + "_" + int_to_str(k+1);
                    this->t[index].y = labels[label];

                    label = int_to_str(i+1) + "_" + int_to_str(j) + "_" + int_to_str(k+1);
                    this->t[index].z = labels[label];

                    label = int_to_str(i+1) + "_" + int_to_str(j+1) + "_" + int_to_str(k);
                    this->t[index].w = labels[label];

                    index++;

                    //Second tetrahedron (0,1,1) (1,0,1) (1,1,0) (1,1,1)
                    label = int_to_str(i) + "_" + int_to_str(j+1) + "_" + int_to_str(k+1);
                    this->t[index].x = labels[label];

                    label = int_to_str(i+1) + "_" + int_to_str(j) + "_" + int_to_str(k+1);
                    this->t[index].y = labels[label];

                    label = int_to_str(i+1) + "_" + int_to_str(j+1) + "_" + int_to_str(k);
                    this->t[index].z = labels[label];

                    label = int_to_str(i+1) + "_" + int_to_str(j+1) + "_" + int_to_str(k+1);
                    this->t[index].w = labels[label];

                    index++;

                    //Third tetrahedron (0,0,0) (0,0,1) (0,1,1) (1,0,1)
                    label = int_to_str(i) + "_" + int_to_str(j) + "_" + int_to_str(k);
                    this->t[index].x = labels[label];

                    label = int_to_str(i) + "_" + int_to_str(j) + "_" + int_to_str(k+1);
                    this->t[index].y = labels[label];

                    label = int_to_str(i) + "_" + int_to_str(j+1) + "_" + int_to_str(k+1);
                    this->t[index].z = labels[label];

                    label = int_to_str(i+1) + "_" + int_to_str(j) + "_" + int_to_str(k+1);
                    this->t[index].w = labels[label];

                    index++;

                    //Fourth tetrahedron (0,0,0) (0,1,0) (0,1,1) (1,1,0)
                    label = int_to_str(i) + "_" + int_to_str(j) + "_" + int_to_str(k);
                    this->t[index].x = labels[label];

                    label = int_to_str(i) + "_" + int_to_str(j+1) + "_" + int_to_str(k);
                    this->t[index].y = labels[label];

                    label = int_to_str(i) + "_" + int_to_str(j+1) + "_" + int_to_str(k+1);
                    this->t[index].z = labels[label];

                    label = int_to_str(i+1) + "_" + int_to_str(j+1) + "_" + int_to_str(k);
                    this->t[index].w = labels[label];

                    index++;

                    //Fifth tetrahedron (0,0,0) (1,0,0) (1,0,1) (1,1,0)
                    label = int_to_str(i) + "_" + int_to_str(j) + "_" + int_to_str(k);
                    this->t[index].x = labels[label];

                    label = int_to_str(i+1) + "_" + int_to_str(j) + "_" + int_to_str(k);
                    this->t[index].y = labels[label];

                    label = int_to_str(i+1) + "_" + int_to_str(j) + "_" + int_to_str(k+1);
                    this->t[index].z = labels[label];

                    label = int_to_str(i+1) + "_" + int_to_str(j+1) + "_" + int_to_str(k);
                    this->t[index].w = labels[label];

                    index++;
                }
            }
        }
    }

    labels.clear();
}//void tetramesh::tetrahedralize(const aabb & bounds, ...)


/// Import a simplicial complex from tetgen .nodes .faces
void tetramesh::import_tetgen (const string &nodeFilename)
{
    char dummy_buffer[1024];

    //Remove the extension of the file
    std::string rootFilename(nodeFilename);
    size_t extPos = rootFilename.rfind(".node");
    if (extPos > 0)
        rootFilename.erase(extPos);

    //Open .node file
    FILE *nodeFile = fopen(nodeFilename.c_str(), "r");
    if (!nodeFile)
    {
        throw nodeFilename + string(": File not found");
    }

    //Read the number of vertex
    int nv;
    if (fscanf(nodeFile, " %d", &nv) != 1)
        throw nodeFilename + string(": error loading file");
    if (fgets(dummy_buffer, 1024, nodeFile) == NULL)
        throw nodeFilename + string(": error loading file");

    this->clear();
    this->x.reserve(nv);

    //Read the values of vertex
    for (int i=0; i<nv ; i++)
    {
        int n;
        float x, y, z;
        //Read values for index, X, Y, Z
        if (fscanf(nodeFile, " %d %f %f %f", &n, &x, &y, &z) != 4)
            throw nodeFilename + string(": error loading file");
        if (fgets(dummy_buffer, 1024, nodeFile)==NULL)
            throw nodeFilename + string(": error loading file");

        this->x.push_back(vec3(x,y,z));
        //std::cout << i << "/" << nv << ":" << x << " " << y << " " << z << std::endl;
    }

    //close the file
    fclose(nodeFile);

    //Open .ele file
    std::string eleFilename(rootFilename);
    eleFilename.append(".ele");
    FILE *eleFile = fopen(eleFilename.c_str(), "r");
    if (!eleFile)
    {
        throw eleFilename + string(": File not found");
    }

    //Read the number of tetraedra
    int nt;
    if (fscanf(eleFile, " %d", &nt) != 1)
        throw eleFilename + string(": Error reading file");
    if (fgets(dummy_buffer, 1024, eleFile) == NULL)
        throw eleFilename + string(": Error reading file");

    this->t.reserve(nt);

    //Read the values of tetra
    for (int i=0; i<nt ; i++)
    {
        int n, a, b, c, d;
        if (fscanf(eleFile, " %d %d %d %d %d", &n, &a, &b, &c, &d) != 5)
            throw eleFilename + string(": Error reading file");
        if (fgets(dummy_buffer, 1024, eleFile)==NULL)
            throw eleFilename + string(": Error reading file");

        this->t.push_back(uvec4(a,b,c,d));
    }

    //Close the file
    fclose(eleFile);

    //Create info for edges
    this->update_edges();

    cout << "Read " << x.size() << " vertex. " << t.size() << " tetras. " << e.size() << " edges. " <<endl;

}//void tetramesh::import_tetgen (const string &nodeFilename)

aabb tetramesh::get_aabb() const {
    aabb result;
    result.set(this->x);
    return result;
}


void tetramesh::clear() {
    this->x.clear();
    this->t.clear();
    this->e.clear();
}


vec3 tetramesh::get_centroid() {
    vec3 centroid(0.0, 0.0, 0.0);
    for (unsigned i=0; i<this->x.size(); i++) {
        centroid += this->x[i];
    }
    double inv_div = 1.0 / double(this->x.size());
    centroid.x *= inv_div;
    centroid.y *= inv_div;
    centroid.z *= inv_div;
    return centroid;
}





//Tester of edges in order to prevent duplicates, used in "mesh::update_edges()".
struct edge_tester {
    bool operator()(const uvec2 s1, const uvec2 s2) const {
        return ((s1.x < s2.x) || (s1.x == s2.x && s1.y < s2.y));
    }
};



void tetramesh::update_edges() {

    set<uvec2, edge_tester> anti_repeats;
    set<uvec2, edge_tester>::iterator it;
    uvec2 edge;
    anti_repeats.clear();
    this->e.clear();

    for (vector<uvec4>::iterator it = this->t.begin(); it != this->t.end(); it++) {
        edge.x = min(it->x, it->y);
        edge.y = max(it->x, it->y);
        anti_repeats.insert(edge);

        edge.x = min(it->x, it->z);
        edge.y = max(it->x, it->z);
        anti_repeats.insert(edge);

        edge.x = min(it->x, it->w);
        edge.y = max(it->x, it->w);
        anti_repeats.insert(edge);

        edge.x = min(it->y, it->z);
        edge.y = max(it->y, it->z);
        anti_repeats.insert(edge);

        edge.x = min(it->y, it->w);
        edge.y = max(it->y, it->w);
        anti_repeats.insert(edge);

        edge.x = min(it->z, it->w);
        edge.y = max(it->z, it->w);
        anti_repeats.insert(edge);
    }

    //~ cout << "There are " << anti_repeats.size() << " unique edges.\n";
    if (anti_repeats.size() > 0) {
        this->e.assign(anti_repeats.begin(), anti_repeats.end());
        anti_repeats.clear();
    }
}



void dilate(vec3 nodes[4], double factor) {
    vec3 centroid;
    centroid.x = (nodes[0].x + nodes[1].x + nodes[2].x + nodes[3].x) * 0.25;
    centroid.y = (nodes[0].y + nodes[1].y + nodes[2].y + nodes[3].y) * 0.25;
    centroid.z = (nodes[0].z + nodes[1].z + nodes[2].z + nodes[3].z) * 0.25;

    for (unsigned i=0; i<4; i++) {
        sub(nodes[i], nodes[i], centroid);
        mul(nodes[i], factor, nodes[i]);
        add(nodes[i], nodes[i], centroid);
    }
}






void create_ibox_grid(iaabb bb,
                      uvec3 box_size,
                      vector<ivec3> & result_nodes,
                      vector<uvec8> & result_boxes) {

    uvec3 steps;
    steps.x = CEIL_POS(float(bb.width()) / float(box_size.x));
    steps.y = CEIL_POS(float(bb.height()) / float(box_size.y));
    steps.z = CEIL_POS(float(bb.depth()) / float(box_size.z));

    result_boxes.clear();
    result_boxes.reserve(steps.x * steps.y * steps.z);

    result_nodes.clear();
    result_nodes.reserve((steps.x+1) * (steps.y+1) * (steps.z+1));

    //Creating the "grid_nodes"
    ivec3 p;

    for (unsigned k = 0; k <= steps.z; k++) {
        p.z = MIN(bb.aabb_min.z + int(box_size.z * k), bb.aabb_max.z);

        for (unsigned j=0; j <= steps.y; j++) {
            p.y = MIN(bb.aabb_min.y + int(box_size.y * j), bb.aabb_max.y);

            for (unsigned i=0; i <= steps.x; i++) {
                p.x = MIN(bb.aabb_min.x + int(box_size.x * i), bb.aabb_max.x);
                result_nodes.push_back(p);
            }
        }
    }

    //Creating the "grid_boxes"
    uvec8 box;

    for (unsigned k = 0; k < steps.z; k++) {
        for (unsigned j=0; j < steps.y; j++) {
            for (unsigned i=0; i < steps.x; i++) {

                box[0] = (k * (steps.y + 1) + j) * (steps.x + 1) + i;
                box[1] = box[0] + 1;

                box[2] = box[0] + (steps.x + 1);
                box[3] = box[2] + 1;

                box[4] = box[0] + (steps.x + 1) * (steps.y + 1);
                box[5] = box[4] + 1;

                box[6] = box[4] + (steps.x + 1);
                box[7] = box[6] + 1;

                result_boxes.push_back(box);
            }
        }
    }
/*
    //Debug: Grid box:
    cout << "Debug: Grid Box of: ("<< bb.aabb_min.x <<", "<< bb.aabb_min.y <<", "<< bb.aabb_min.z <<")x";
    cout << "("<< bb.aabb_max.x <<", "<< bb.aabb_max.y <<", "<< bb.aabb_max.z <<")\n";

    cout << "Nodes: ["<< result_nodes.size() <<"] grid of "<< steps.x+1 <<"x"<< steps.y+1 <<"x"<< steps.z+1 <<"\n";
    for (unsigned i=0; i<result_nodes.size(); i++) {
        p = result_nodes[i];
        cout << "["<< i <<"] ("<< p.x <<", "<< p.y <<", "<< p.z <<")\n";
    }

    cout << "Boxes: ["<< result_boxes.size() <<"] grid of "<< steps.x <<"x"<< steps.y <<"x"<< steps.z <<"\n";
    for (unsigned i=0; i<result_boxes.size(); i++) {
        box = result_boxes[i];
        cout << "["<< i <<"]";
        for (unsigned j=0; j<8; j++) {
            cout << " " << box[j];
        }
        cout << endl;
    }
*/
}
