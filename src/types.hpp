
#ifndef _TYPES_HH_
#define _TYPES_HH_

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <set>
#include <string>
#include <math.h>
#include <map>
#include <assert.h>
#include <iostream>


using namespace std;


string replace (string entrada, string busca, string cambia);

string int_to_str (int valor);

int str_to_int (string valor);

string double_to_str (double valor);

double str_to_double (string valor);

string float_to_str (float valor);

string to_binary_format (const unsigned char v);




#define MIN(X,Y) (X < Y ? X : Y)
#define MAX(X,Y) (X > Y ? X : Y)

#define CEIL_POS(X) ((X-(int)(X)) > 0 ? (int)(X+1) : (int)(X))
#define CEIL_NEG(X) ((X-(int)(X)) < 0 ? (int)(X-1) : (int)(X))
#define CEIL(X) (((X) > 0) ? CEIL_POS(X) : CEIL_NEG(X))

#define FLOOR_POS(X) ((X-(int)(X)) < 0 ? (int)(X-1) : (int)(X))
#define FLOOR_NEG(X) ((X-(int)(X)) < 0 ? (int)(X-1) : (int)(X))
#define FLOOR(X) (((X) > 0) ? FLOOR_POS(X) : FLOOR_NEG(X))

inline unsigned char lerp(const unsigned char a, const unsigned char b, const float alpha) {
    //return a * alpha + b * (1.0 - alpha);
    return (unsigned char) ((a-b) * alpha + b);
}

inline short lerp(const short a, const short b, const float alpha) {
    //return a * alpha + b * (1.0 - alpha);
    return short((a-b) * alpha + b);
}

inline short lerp2(const short a, const short b, const float alpha, const float beta) {
    return short(a * alpha + b * beta);
}

#ifndef round
#define round(x) (floor(x+0.5))
#endif



class vec2 {
public:
    double x,y;

    vec2(double _x = 0, double _y = 0);

    double dot(const vec2 & op1, const vec2 & op2);
    void set(double _x, double _y);
    void set(const vec2 & v);
    void normalize();
    double length();
    double length2();


    double & operator [] (const unsigned int i) {
        switch(i) {
            case 0: return this->x;
            default: return this->y;
        }
    }


    double operator [] (const unsigned int i) const {
        switch(i) {
            case 0: return this->x;
            default: return this->y;
        }
    }


    bool operator == (const vec2 &o) const {
        return (fabs(this->x - o.x) < 0.000001 &&
                fabs(this->y - o.y) < 0.000001);
    }


    bool operator != (const vec2 o) {
        return (fabs(this->x - o.x) > 0.000001 ||
                fabs(this->y - o.y) > 0.000001);
    }


    void operator = (const vec2 &o) {
        this->x = o.x;
        this->y = o.y;
    }


    vec2 operator + (const vec2 &o) const {
        vec2 r = *this;
        r.x += o.x;
        r.y += o.y;
        return r;
    }


    vec2 operator - (const vec2 &o) const {
        vec2 r = *this;
        r.x -= o.x;
        r.y -= o.y;
        return r;
    }

    void operator += (const vec2 &o) {
        this->x += o.x;
        this->y += o.y;
    }


    void operator -= (const vec2 &o) {
        this->x -= o.x;
        this->y -= o.y;
    }

    ///Scale a vec2 by another vec
    inline vec2 scale(const vec2 &op) const
    {
        return vec2(x * op.x, y * op.y);
    }

    ///Scale a vec2 by the inverse of another vec
    inline vec2 scaleInv(const vec2 &op) const
    {
        assert (op.x != 0 && op.y != 0);
        return vec2(x / op.x, y / op.y);
    }

    string to_s();
};


void mul(vec2 * r, double v, const vec2 * op);

void mul(vec2 & r, double v, const vec2 & op);

double dot(vec2 * op1, vec2 * op2);


void add(vec2 * r, const vec2 * op1, const vec2 * op2);

void add(vec2 & r, const vec2 & op1, const vec2 & op2);

void sub(vec2 * r, const vec2 * op1, const vec2 * op2);

void sub(vec2 & r, const vec2 & op1, const vec2 & op2);




class vec3 {
public:
    double x,y,z;

    vec3(double _x = 0, double _y = 0, double _z = 0) {
        x=_x; y=_y; z=_z;
    }

    vec3(const double *xyz) {
        x = xyz[0];
        y = xyz[1];
        z = xyz[2];
    }

    double dot(const vec3 & op1) const
    {
        return x * op1.x + y * op1.y + z * op1.z;
    }

    void set(const double *xyz) {
        this->x = xyz[0];
        this->y = xyz[1];
        this->z = xyz[2];
    }

    void set(double _x, double _y, double _z) {
        this->x = _x;
        this->y = _y;
        this->z = _z;
    }

    void set(const vec3 & v) {
        this->x = v.x;
        this->y = v.y;
        this->z = v.z;
    }

    void normalize() {
        double length = this->x * this->x +
                             this->y * this->y +
                             this->z * this->z;
        if (length < 0.00000001) {
            return;
        }
        length = 1.0 / sqrt(length);
        this->x *= length;
        this->y *= length;
        this->z *= length;
    }

    double length() const {
        return sqrt(this->x * this->x +
                    this->y * this->y +
                    this->z * this->z);
    }


    double length2() const {
        return (this->x * this->x +
                this->y * this->y +
                this->z * this->z);
    }



    double & operator [] (const unsigned int i) {
        switch(i) {
            case 0: return this->x;
            case 1: return this->y;
            default: return this->z;
        }
    }


    vec3  operator * (const vec3 other);

    double operator [] (const unsigned int i) const {
        switch(i) {
            case 0: return this->x;
            case 1: return this->y;
            default: return this->z;
        }
    }


    bool operator == (const vec3 &o) const {
        return (fabs(this->x - o.x) < 0.000001 &&
                fabs(this->y - o.y) < 0.000001 &&
                fabs(this->z - o.z) < 0.000001);
    }


    bool operator != (const vec3 o) {
        return (fabs(this->x - o.x) > 0.000001 ||
                fabs(this->y - o.y) > 0.000001 ||
                fabs(this->z - o.z) > 0.000001);
    }


    void operator = (const vec3 &o) {
        this->x = o.x;
        this->y = o.y;
        this->z = o.z;
    }


    vec3 operator + (const vec3 &o) const {
        vec3 r = *this;
        r.x += o.x;
        r.y += o.y;
        r.z += o.z;
        return r;
    }


    vec3 operator - (const vec3 &o) const {
        vec3 r = *this;
        r.x -= o.x;
        r.y -= o.y;
        r.z -= o.z;
        return r;
    }

    void operator += (const vec3 &o) {
        this->x += o.x;
        this->y += o.y;
        this->z += o.z;
    }


    void operator -= (const vec3 &o) {
        this->x -= o.x;
        this->y -= o.y;
        this->z -= o.z;
    }

    ///Scale a vec3 by another vec
    inline vec3 scale(const vec3 &op) const
    {
        return vec3(x * op.x, y * op.y, z * op.z);
    }

    ///Scale a vec3 by the inverse of another vec
    inline vec3 scaleInv(const vec3 &op) const
    {
        assert (op.x != 0 && op.y != 0 && op.z != 0);
        return vec3(x / op.x, y / op.y, z / op.z);
    }

    inline double distance(const vec3 &v) const
    {
        return sqrt((x-v.x)*(x-v.x)+(y-v.y)*(y-v.y)+(z-v.z)*(z-v.z));
    }

    inline double distance2(const vec3 &v) const
    {
        return (x-v.x)*(x-v.x)+(y-v.y)*(y-v.y)+(z-v.z)*(z-v.z);
    }

    string to_s() const;
};


void mul(vec3 * r, double v, const vec3 * op);

void mul(vec3 & r, double v, const vec3 & op);

double dot(vec3 * op1, vec3 * op2);

void cross(vec3 * r, const vec3 * op1, const vec3 * op2);


void add(vec3 * r, const vec3 * op1, const vec3 * op2);

void add(vec3 & r, const vec3 & op1, const vec3 & op2);

void sub(vec3 * r, const vec3 * op1, const vec3 * op2);

void sub(vec3 & r, const vec3 & op1, const vec3 & op2);


class uvec2 {
public:
    unsigned x,y;

    uvec2(unsigned _x = 0, unsigned _y = 0);
    unsigned length2();
    void set(unsigned _x, unsigned _y);
    void set(const uvec2 & o);
    string to_s();

    bool operator == (const uvec2 o) {
        return (this->x == o.x && this->y == o.y);
    }

    bool operator != (const uvec2 o) {
        return (this->x != o.x || this->y != o.y);
    }

    unsigned & operator [] (const unsigned i) {
        switch(i) {
            case 0: return this->x;
            default: return this->y;
        }
    }

    unsigned operator [] (const unsigned i) const {
        switch(i) {
            case 0: return this->x;
            default: return this->y;
        }
    }

};



class ivec3 {
public:
    int x,y,z;

    ivec3(int _x = 0, int _y = 0, int _z = 0);
    int length2();
    void set(int _x, int _y, int _z);
    void set(const ivec3 & o);
    string to_s();

    int & operator [] (const unsigned i) {
        switch(i) {
            case 0: return this->x;
            case 1: return this->y;
            default: return this->z;
        }
    }

    int operator [] (const unsigned i) const {
        switch(i) {
            case 0: return this->x;
            case 1: return this->y;
            default: return this->z;
        }
    }

    bool operator == (const ivec3 &o) const {
        return (this->x == o.x && this->y == o.y && this->z == o.z);
    }


    bool operator != (const ivec3 &o) const {
        return (this->x != o.x || this->y != o.y || this->z != o.z);
    }


    void operator = (const ivec3 o) {
        this->x = o.x;
        this->y = o.y;
        this->z = o.z;
    }


    void operator = (const vec3 o) {
        this->x = int(round(o.x));
        this->y = int(round(o.y));
        this->z = int(round(o.z));
    }


    ivec3 operator + (const ivec3 o) const {
        ivec3 r = *this;
        r.x += o.x;
        r.y += o.y;
        r.z += o.z;
        return r;
    }


    ivec3 operator + (const vec3 o) const {
        ivec3 r = *this;
        r.x += int(round(o.x));
        r.y += int(round(o.x));
        r.z += int(round(o.x));
        return r;
    }



    ivec3 operator - (const ivec3 o) const {
        ivec3 r = *this;
        r.x -= o.x;
        r.y -= o.y;
        r.z -= o.z;
        return r;
    }

    void operator += (const ivec3 o) {
        this->x += o.x;
        this->y += o.y;
        this->z += o.z;
    }


    void operator -= (const ivec3 o) {
        this->x -= o.x;
        this->y -= o.y;
        this->z -= o.z;
    }



    void operator += (const vec3 o) {
        this->x += int(round(o.x));
        this->y += int(round(o.y));
        this->z += int(round(o.z));
    }


    void operator -= (const vec3 o) {
        this->x -= int(round(o.x));
        this->y -= int(round(o.y));
        this->z -= int(round(o.z));
    }

};

void sub(ivec3 & r, const ivec3 & op1, const ivec3 & op2);

void add(ivec3 & r, const ivec3 & op1, const ivec3 & op2);



class uvec3 {
public:
    unsigned x,y,z;

    uvec3(unsigned _x = 0, unsigned _y = 0, unsigned _z = 0);

    unsigned & operator [] (const unsigned i) {
        switch(i) {
            case 0: return this->x;
            case 1: return this->y;
            default: return this->z;
        }
    }

    unsigned operator [] (const unsigned i) const {
        switch(i) {
            case 0: return this->x;
            case 1: return this->y;
            default: return this->z;
        }
    }

    bool operator == (const uvec3 &o) const {
        return (this->x == o.x && this->y == o.y && this->z == o.z);
    }

    bool operator != (const uvec3 &o) const {
        return (this->x != o.x || this->y != o.y || this->z == o.z);
    }

    string to_s();

    void set(unsigned _x, unsigned _y, unsigned _z);
};



class vec4 {
public:
    double x,y,z,w;

    vec4(double _x = 0, double _y = 0, double _z = 0, double _w = 0);

    void set(double _x, double _y, double _z, double _w);

    void set(const vec4 & v);

    void set(const vec3 & v);

    double & operator [] (const unsigned int i) {
        switch(i) {
            case 0: return this->x;
            case 1: return this->y;
            case 2: return this->z;
            default: return this->w;
        }
    }

    double operator [] (const unsigned int i) const {
        switch(i) {
            case 0: return this->x;
            case 1: return this->y;
            case 2: return this->z;
            default: return this->w;
        }
    }

    string to_s();
};


class ivec4 {
public:
    int x,y,z,w;

    ivec4(int _x = 0, int _y = 0, int _z = 0, int _w = 0);
    void set(int _x, int _y, int _z, int _w);
    string to_s();
};


class uvec4 {
public:
    unsigned x,y,z,w;

    uvec4(unsigned _x = 0, unsigned _y = 0, unsigned _z = 0, unsigned _w = 0);

    void set(unsigned _x, unsigned _y, unsigned _z, unsigned _w);

    unsigned & operator [] (const unsigned int i) {
        switch(i) {
            case 0: return this->x;
            case 1: return this->y;
            case 2: return this->z;
            default: return this->w;
        }
    }

    unsigned operator [] (const unsigned int i) const {
        switch(i) {
            case 0: return this->x;
            case 1: return this->y;
            case 2: return this->z;
            default: return this->w;
        }
    }

    string to_s();
};



class aabb {
    public:
    vec3 aabb_min;
    vec3 aabb_max;

    aabb(double min_x = 0.0, double min_y = 0.0, double min_z = 0.0,
         double max_x = 0.0, double max_y = 0.0, double max_z = 0.0);

    aabb(const vector<vec3> & v);

    ~aabb();

    vec3 midpoint();

    ///Get the 3D coordinates of the ith corner of the aabb.
    ///The aabb has 8 corners, and the first is "aabb_min" and the last "aabb_max".
    vec3 get_node_th(unsigned i);

    void set(const vec3 & v);
    void add(const vec3 & v);

    void set(const vector<vec3> & v);
    void add(const vector<vec3> & v);

    void set(const vec3 v[4]);
    void set(const double x, const double y, const double z);

    //Return the width, height or depth of the aabb
    double width() const;
    double height() const;
    double depth() const;

    unsigned iwidth() const;
    unsigned iheight() const;
    unsigned idepth() const;
    //Return a vec3 with the width, height and depth of the aabb.
    vec3 dim() const;
    uvec3 udim() const;
    unsigned size() const;

    bool is_outside(aabb & other);

    inline bool is_inside(const vec3 & other) const {
        return (this->is_inside(other.x, other.y, other.z));
    }

    inline bool is_inside(const double x, const double y, const double z) const {
        return (this->aabb_min.x <= x &&
                this->aabb_min.y <= y &&
                this->aabb_min.z <= z &&
                this->aabb_max.x >= x &&
                this->aabb_max.y >= y &&
                this->aabb_max.z >= z);
    }


    inline bool is_outside(const vec3 & other) const {
        return (!this->is_inside(other));
    }

    ///Return true if a point is near of the interface of an aabb.
    inline bool is_near(const double x, const double y, const double z, double max_distance) const {
        return (fabs(x - this->aabb_min.x) <= max_distance ||
                fabs(y - this->aabb_min.y) <= max_distance ||
                fabs(z - this->aabb_min.z) <= max_distance ||
                fabs(x - this->aabb_max.x) <= max_distance ||
                fabs(y - this->aabb_max.y) <= max_distance ||
                fabs(z - this->aabb_max.z) <= max_distance);
    }

    void adapt(double new_size_x, double new_size_y, double new_size_z);
    void clear();
    string to_s();
};




class iaabb {
    public:
    ivec3 aabb_min;
    ivec3 aabb_max;

    iaabb(int min_x = 0.0, int min_y = 0.0, int min_z = 0.0,
         int max_x = 0.0, int max_y = 0.0, int max_z = 0.0);

    ~iaabb();

    void set(int min_x = 0.0, int min_y = 0.0, int min_z = 0.0,
             int max_x = 0.0, int max_y = 0.0, int max_z = 0.0);

    void set(const ivec3 & v);
    void set(const vector<ivec3> & v);
    void add(const vector<ivec3> & v);
    void add(const ivec3 & v);

    //Return the width, height or depth of the aabb
    int width();
    int height();
    int depth();
    //Return a vec3 with the width, height and depth of the aabb.
    uvec3 dim();

    void adapt(int new_size_x, int new_size_y, int new_size_z);
    void clear();
    string to_s();
};




class trimesh {
    public:
    trimesh();

    ~trimesh();

    //List of vertices.
    vector<vec3> x;
    //List of triangles (faces).
    vector<uvec3> f;
    //List of normals.
    vector<vec3> n;
    //List of uv coordinates.
    vector<vec2> uv;

    aabb get_aabb();

    void clear();

    void load(string filename);

    vector<unsigned> tex_ids;

    unsigned _shader;
};



class tetramesh {
public:
    tetramesh();

    ~tetramesh();

    //List of vertices.
    vector<vec3> x;
    //List of tetrahedra.
    vector<uvec4> t;
    //List of edges.
    vector<uvec2> e;

    void tetrahedralize(const aabb &bounds, unsigned divX, unsigned divY = 0, unsigned divZ = 0);

    /// Read a tetrahedral complex in tetgen file format (.node + .ele files)
    void import_tetgen (const string &nodeFilename);

    aabb get_aabb() const;

    void update_edges();

    void clear();

    vec3 get_centroid();

};



class uvec8 {
    public:
    unsigned v[8];

    unsigned & operator [] (const unsigned i) { return v[i]; }
    unsigned operator [] (const unsigned i) const { return v[i]; }
};



//Given the four nodes of a tetrahedron, a little dilation is perfomed on them.
void dilate(vec3 nodes[4], double factor);


//Given a bounding box, this method creates a grid of nodes and boxes.
void create_ibox_grid(iaabb bb,
                      uvec3 box_size,
                      vector<ivec3> & result_nodes,
                      vector<uvec8> & result_boxes);

#endif //_TYPES_HH_

