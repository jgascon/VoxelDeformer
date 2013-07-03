//
// C++ Interface: matrix
//
// Description:
//
//
// Author: Jorge Gascon Perez, <jorge@IronMaiden>, (C) 2009
//
// Copyright: See COPYING file that comes with this distribution
//
//

#ifndef __MATRIX__H__
#define __MATRIX__H__

#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <types.hpp>

#define PI 3.14159265
#define DEG_TO_RAD PI/180


using namespace std;

class matrix {

    protected:

        unsigned _rows;

        unsigned _cols;

        ///Internal storage of information.
        vector<float> _local;

        ///Pointer to the active array of information this matrix stores, is
        ///a pointer in case to be mapped to external data.
        float * _data;

        unsigned _prev_rows;

        unsigned _prev_cols;

    public:

        ///Empty constructor.
        matrix();

        matrix(unsigned rows, unsigned cols);

        matrix(unsigned rows, unsigned cols, const vector<float> & data);

        matrix(unsigned rows, unsigned cols, float * data);

        ///Creates a 1x3 vector.
        matrix(float x, float y, float z);

        ///Creates a matrix which is a copy of a given matrix.
        matrix(const matrix & original);

        ~matrix();

        void clear();

        unsigned rows() const;

        unsigned cols() const;

        //~ vector<float> & data();

        float * data();

        const float * data() const;

        //~ const vector<float> & data() const;

        void set_identity();

        bool is_identity();

        ///Returns true if the matrix is full of zeroes.
        bool is_zero();

        ///Sets all the positions with the given value (zero by default).
        void set(float v = 0.0);

        void set(const matrix & v);

        ///Make a copy of our matrix and puts it in v.
        void clone(matrix & v);

        void resize(unsigned rows, unsigned cols);

        void resize(unsigned rows, unsigned cols, const vector<float> & data);

        void resize(unsigned rows, unsigned cols, const float * data);

        inline float * operator [] (const unsigned i) {
            assert (i< this->_rows);
            return &this->_data[i * this->_cols];
        }

        inline const float * operator [] (const unsigned i) const {
            assert (i< this->_rows);
            return &this->_data[i * this->_cols];
        }

        string to_s();

        void transpose();

        //Returns the determinant of given matrix, or zero.
        float det();

        bool invert();

        bool ortonormalize();

        bool is_symetric();

        ///Factorizes the matrix using Choleski, returns a lower triangular matrix.
        bool factorize();

        ///Change the sign of each component.
        void negate();

        void operator += (matrix & other);

        void operator -= (matrix & other);

        //~ void operator *= (matrix & other);

        bool operator == (const matrix & other) const;

        bool operator != (const matrix & other) const;

        ///Returns true if all the values in the main diagonal are different to zero.
        bool is_defined_diag() const;

        ///Returns the sum of all values of the matrix.
        float summatory() const;

        ///Returns the sum of all absolute values of the matrix.
        float abs_summatory() const;

        void set_submatrix(matrix & m);

        void set_submatrix(unsigned rows, unsigned cols, matrix & m);

        void map(unsigned rows, unsigned cols, float * external);

        void unmap();

        vec3 operator * (const vec3 & op) const {
            vec3 r(0.0, 0.0, 0.0);
            if (this->cols() == 3) {
                for (unsigned i=0; i<3; i++) {
                    r[i] = (*this)[i][0] * op[0] + (*this)[i][1] * op[1] + (*this)[i][2] * op[2];
                }
            }
            return r;
        }

        vec3 operator * (const ivec3 & op) const {
            vec3 op_1(op.x, op.y, op.z);
            return *this * op_1;
        }

};


bool add(matrix & result, const matrix & op1, const matrix & op2);

bool mul(matrix & result, const matrix & op1, const matrix & op2);

bool mul(matrix & result, const float v, const matrix & op);

bool accum(matrix & result, const matrix & op1, const matrix & op2);

bool sub(matrix & result, const matrix & op1, const matrix & op2);

void fmul3x3(vec3 & r, const float * op1, const vec3 & op2);

inline void mul(vec3 & result, const matrix & op1, const vec3 & op2) {
    assert(op1.cols() == 3);
    for (unsigned i=0; i<3; i++) {
        result[i] = op1[i][0] * op2[0] + op1[i][1] * op2[1] + op1[i][2] * op2[2];
    }
}//inline void mul(vec3 & result, const matrix & op1, const vec3 & op2)

bool mul(vec4 & result, const matrix & op1, const vec4 & op2);


///Adds a block of a matrix (in a given offset) to the block of another.
bool block_add(unsigned block_rows, unsigned block_cols,
               matrix & dest, unsigned dest_offset_row, unsigned dest_offset_col,
               const matrix & orig, unsigned orig_offset_row, unsigned orig_offset_col);

bool block_add(matrix & dest, unsigned dest_offset_row, unsigned dest_offset_col,
               const matrix & orig);

bool block_set(unsigned block_rows, unsigned block_cols,
               matrix & dest, unsigned dest_offset_row, unsigned dest_offset_col,
               const matrix & orig, unsigned orig_offset_row, unsigned orig_offset_col);

bool block_set(matrix & dest, unsigned dest_offset_row, unsigned dest_offset_col,
               const matrix & orig);


///Applies a preconditioner in order to increase the performance of
///the solve_gs and solve_gc methods.
void apply_preconditioner(matrix & A, matrix & b);

///Solve the linear system Ax = b using Gauss-Seidel
bool solve_gs(const matrix & A, matrix & x, const matrix & b);

///Solve the linear system Ax = b using Conjugate Gradient
bool solve_gc(const matrix & A, matrix & x, const matrix & b);

///Solve the linear system Ax = b using Choleski, this method needs the A matrix
///to be lower triangular (factorized by choleski method).
bool solve_choleski(const matrix & A, matrix & x, const matrix & b);

bool solve_choleski(const matrix & A, const matrix & A_t, matrix & x, const matrix & b);

bool solve_choleski(const matrix & A, const matrix & d, const matrix & A_t, matrix & x, const matrix & b);


matrix create_rotation_matrix_around_X(float degrees);

matrix create_rotation_matrix_around_Y(float degrees);

matrix create_rotation_matrix_around_Z(float degrees);







class fmatrix {

    protected:

        unsigned _rows;

        unsigned _cols;

        ///Internal storage of information.
        vector<float> _local;

        ///Pointer to the active array of information this fmatrix stores, is
        ///a pointer in case to be mapped to external data.
        float * _data;

        unsigned _prev_rows;

        unsigned _prev_cols;

    public:

        ///Empty constructor.
        fmatrix();

        fmatrix(unsigned rows, unsigned cols);

        fmatrix(unsigned rows, unsigned cols, const vector<float> & data);

        fmatrix(unsigned rows, unsigned cols, float * data);

        ///Creates a 1x3 vector.
        fmatrix(float x, float y, float z);

        ///Creates a fmatrix which is a copy of a given fmatrix.
        fmatrix(const fmatrix & original);

        ~fmatrix();

        void clear();

        unsigned rows() const;

        unsigned cols() const;

        //~ vector<float> & data();

        float * data();

        const float * data() const;

        //~ const vector<float> & data() const;

        void set_identity();

        bool is_identity();

        ///Returns true if the fmatrix is full of zeroes.
        bool is_zero();

        ///Sets all the positions with the given value (zero by default).
        void set(float v = 0.0);

        void set(const fmatrix & v);

        ///Make a copy of our fmatrix and puts it in v.
        void clone(fmatrix & v);

        void resize(unsigned rows, unsigned cols);

        void resize(unsigned rows, unsigned cols, const vector<float> & data);

        void resize(unsigned rows, unsigned cols, const float * data);

        inline float * operator [] (const unsigned i) {
            assert (i< this->_rows);
            return &this->_data[i * this->_cols];
        }

        inline const float * operator [] (const unsigned i) const {
            assert (i< this->_rows);
            return &this->_data[i * this->_cols];
        }

        string to_s();

        void transpose();

        //Returns the determinant of given fmatrix, or zero.
        float det();

        bool invert();

        bool ortonormalize();

        bool is_symetric();

        ///Factorizes the fmatrix using Choleski, returns a lower triangular fmatrix.
        bool factorize();

        ///Change the sign of each component.
        void negate();

        void operator += (fmatrix & other);

        void operator -= (fmatrix & other);

        //~ void operator *= (fmatrix & other);

        bool operator == (const fmatrix & other) const;

        bool operator != (const fmatrix & other) const;

        ///Returns true if all the values in the main diagonal are different to zero.
        bool is_defined_diag() const;

        ///Returns the sum of all values of the fmatrix.
        float summatory() const;

        ///Returns the sum of all absolute values of the fmatrix.
        float abs_summatory() const;

        void set_subfmatrix(fmatrix & m);

        void set_subfmatrix(unsigned rows, unsigned cols, fmatrix & m);

        void map(unsigned rows, unsigned cols, float * external);

        void unmap();

        vec3 operator * (const vec3 & op) const {
            vec3 r(0.0, 0.0, 0.0);
            if (this->cols() == 3) {
                for (unsigned i=0; i<3; i++) {
                    r[i] = (*this)[i][0] * op[0] + (*this)[i][1] * op[1] + (*this)[i][2] * op[2];
                }
            }
            return r;
        }

        vec3 operator * (const ivec3 & op) const {
            vec3 op_1(op.x, op.y, op.z);
            return *this * op_1;
        }

};


bool add(fmatrix & result, const fmatrix & op1, const fmatrix & op2);

bool mul(fmatrix & result, const fmatrix & op1, const fmatrix & op2);

bool mul(fmatrix & result, const float v, const fmatrix & op);

bool accum(fmatrix & result, const fmatrix & op1, const fmatrix & op2);

bool sub(fmatrix & result, const fmatrix & op1, const fmatrix & op2);


inline void mul(vec3 & result, const fmatrix & op1, const vec3 & op2) {
    assert(op1.cols() == 3);
    for (unsigned i=0; i<3; i++) {
        result[i] = op1[i][0] * op2[0] + op1[i][1] * op2[1] + op1[i][2] * op2[2];
    }
}//inline void mul(vec3 & result, const fmatrix & op1, const vec3 & op2)

bool mul(vec4 & result, const fmatrix & op1, const vec4 & op2);


///Adds a block of a fmatrix (in a given offset) to the block of another.
bool block_add(unsigned block_rows, unsigned block_cols,
               fmatrix & dest, unsigned dest_offset_row, unsigned dest_offset_col,
               const fmatrix & orig, unsigned orig_offset_row, unsigned orig_offset_col);

bool block_add(fmatrix & dest, unsigned dest_offset_row, unsigned dest_offset_col,
               const fmatrix & orig);

bool block_set(unsigned block_rows, unsigned block_cols,
               fmatrix & dest, unsigned dest_offset_row, unsigned dest_offset_col,
               const fmatrix & orig, unsigned orig_offset_row, unsigned orig_offset_col);

bool block_set(fmatrix & dest, unsigned dest_offset_row, unsigned dest_offset_col,
               const fmatrix & orig);


///Applies a preconditioner in order to increase the performance of
///the solve_gs and solve_gc methods.
void apply_preconditioner(fmatrix & A, fmatrix & b);

///Solve the linear system Ax = b using Gauss-Seidel
bool solve_gs(const fmatrix & A, fmatrix & x, const fmatrix & b);

///Solve the linear system Ax = b using Conjugate Gradient
bool solve_gc(const fmatrix & A, fmatrix & x, const fmatrix & b);

///Solve the linear system Ax = b using Choleski, this method needs the A fmatrix
///to be lower triangular (factorized by choleski method).
bool solve_choleski(const fmatrix & A, fmatrix & x, const fmatrix & b);

bool solve_choleski(const fmatrix & A, const fmatrix & A_t, fmatrix & x, const fmatrix & b);

bool solve_choleski(const fmatrix & A, const fmatrix & d, const fmatrix & A_t, fmatrix & x, const fmatrix & b);



#endif //__MATRIX__H__
