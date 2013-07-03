
#include <matrix.hpp>
#include <cstdlib>
#include <string.h>
#include <math.h>
#include <iostream>

using namespace std;

matrix::matrix() {
    this->_data = (float *) this->_local.data();
    this->clear();
}


matrix::matrix(unsigned rows, unsigned cols) {
    this->_local.resize(rows * cols);
    this->_data = (float *) this->_local.data();
    this->_rows = rows;
    this->_cols = cols;
}



matrix::matrix(unsigned rows, unsigned cols, const vector<float> & data) {
    this->_rows = rows;
    this->_cols = cols;

    if (rows == 0 || cols == 0) {
        cout << "Error: matrix::matrix(), a matrix cannot have zero rows or zero cols\n";
        return;
    }

    this->resize(rows, cols, data);
    this->_data = (float *) this->_local.data();
}



matrix::matrix(unsigned rows, unsigned cols, float * data) {
    this->_data = (float *) this->_local.data();
    this->_rows = rows;
    this->_cols = cols;

    if (rows == 0 || cols == 0) {
        cout << "Error: matrix::matrix(), a matrix cannot have zero rows or zero cols\n";
        return;
    }
    if (data == NULL) {
        cout << "Error: matrix::matrix(), data is null\n";
        return;
    }
    this->resize(rows, cols, data);
}



matrix::matrix(float x, float y, float z) {
    this->_data = (float *) this->_local.data();
    this->_rows = 1;
    this->_cols = 3;
    this->_local.resize(3);
    this->_local[0] = x;
    this->_local[1] = y;
    this->_local[2] = z;
}


matrix::matrix(const matrix & original) {
    this->_data = (float *) this->_local.data();
    this->_rows = original.rows();
    this->_cols = original.cols();

    if (_rows == 0 || _cols == 0) {
        cout << "Error: matrix::matrix(), a matrix cannot have zero rows or zero cols\n";
        return;
    }
    this->resize(original.rows(), original.cols(), original.data());
}


matrix::~matrix() {
    this->clear();
}


void matrix::clear() {
    this->_rows = 0;
    this->_cols = 0;

    if (this->_data != this->_local.data()) {
        this->unmap();
    }
    this->_local.clear();
}


unsigned matrix::rows() const {
    return this->_rows;
}


unsigned matrix::cols() const {
    return this->_cols;
}


float * matrix::data() {
    return this->_data;
}


const float * matrix::data() const {
    return this->_data;
}


//~ const vector<float> & matrix::data() const {
    //~ return *this->_data;
//~ }


void matrix::set_identity() {
    for (unsigned i=0; i<this->_rows; i++) {
        for (unsigned j=0; j<this->_cols; j++) {
            if (i == j) {
                this->_data[i*this->_rows + j] = (float) 1.0;
            } else {
                this->_data[i*this->_rows + j] = (float) 0.0;
            }
        }
    }
}



bool matrix::is_identity() {
    if (this->_rows != this->_cols) {
        return false;
    }
    float * p = (float *) this->_data;
    for (unsigned i=0; i<this->_rows; i++) {
        for (unsigned j=0; j<this->_cols; j++) {
            if (i==j) {
                if (*p < 0.99999 || *p > 1.00001) {
                    return false;
                }
            } else {
                if (*p < -0.00001 || *p > 0.00001) {
                    return false;
                }
            }
            p++;
        }
    }
    return true;
}




bool matrix::is_zero() {
    for (unsigned i=0; i<this->_rows * this->_cols; i++) {
        if (this->_data[i] < -0.00001 || this->_data[i] > 0.00001) {
            return false;
        }
    }
    return true;
}



void matrix::set(float v) {
    for (unsigned i=0; i<this->_rows * this->_cols; i++) {
        this->_data[i] = v;
    }
}



void matrix::set(const matrix & v) {
    if (this->_rows != v.rows() || this->_cols != v.cols()) {
        this->resize(v.rows(), v.cols(), v.data());
    } else {
        for (unsigned i=0; i<this->_rows * this->_cols; i++) {
            this->_data[i] = v.data()[i];
        }
    }
}


void matrix::clone(matrix & v) {
    if (this->_rows != v.rows() || this->_cols != v.cols()) {
        v.resize(this->_rows, this->_cols, this->_data);
    } else {
        for (unsigned i=0; i<this->_rows * this->_cols; i++) {
            v.data()[i] = this->_data[i];
        }
    }
}


void matrix::resize(unsigned rows, unsigned cols) {
    if (rows != this->_rows || cols != this->_cols) {
        this->_rows = rows;
        this->_cols = cols;
        if (this->_data == this->_local.data()) {
            //~ cout << "resizing\n";
            this->_local.resize(rows * cols);
            this->_data = (float *) this->_local.data();
        } else {
            cout << "matrix::resize --> Cannot resize mapped data\n";
        }
    }
}


void matrix::resize(unsigned rows, unsigned cols, const vector<float> & data) {
    this->_rows = rows;
    this->_cols = cols;
    if (this->_data == this->_local.data()) {
        this->_local.assign(data.begin(), data.end());
        this->_data = (float *) this->_local.data();
    } else {
        cout << "matrix::resize --> Cannot resize mapped data\n";
    }
}


void matrix::resize(unsigned rows, unsigned cols, const float * data) {
    this->_rows = rows;
    this->_cols = cols;
    if (this->_data == this->_local.data()) {
        this->_local.resize(rows * cols);
        this->_data = (float *) this->_local.data();
        for (unsigned i=0; i<rows*cols; i++) {
            this->_local[i] = data[i];
        }
    } else {
        cout << "matrix::resize --> Cannot resize mapped data\n";
    }
}


string matrix::to_s() {
    if (this->_rows == 0 || this->_cols == 0) {
        return "Empty matrix\n";
    }
    string result = "";

    //cout << "Debug: Real length: " << this->_length << " used length: " << this->_rows * this->_cols << endl;

    for (unsigned i=0; i<this->_rows; i++) {
        for (unsigned j=0; j<this->_cols; j++) {
            if (this->_data[i*this->_cols + j] < 0.00001) {
                result += "  " + float_to_str(this->_data[i*this->_cols + j]);
            } else {
                result += "   " + float_to_str(this->_data[i*this->_cols + j]);
            }
        }
        result += "\n";
    }
    return result;
}


void matrix::transpose() {
    if (this->_rows == 0 || this->_cols == 0) {
        cout << "Error: Transpose --> Cannot transpose an empty matrix.\n";
        return;
    }

    if (this->_cols == 1 || this->_rows == 1) {
        unsigned temp = this->_rows;
        this->_rows = this->_cols;
        this->_cols = temp;
        return;
    }

    vector<float> aux;
    aux.resize(this->_rows * this->_cols, 0.0);

    for (unsigned i=0; i<this->_rows; i++) {
        for (unsigned j=0; j<this->_cols; j++) {
            aux[j * this->_rows + i] = this->_data[i * this->_cols + j];
        }
    }
    unsigned temp = this->_rows;
    this->_rows = this->_cols;
    this->_cols = temp;

    for (unsigned i=0; i<this->_rows*this->_cols; i++) {
        this->_data[i] = aux.data()[i];
    }

    aux.clear();
}



bool matrix::invert() {

    if (this->_rows != this->_cols) {
        cerr << "ERROR: matrix::invert --> Non square matrix\n";
        return false;
    }

    float temp;

    if (this->_rows < 4) {

        float det = this->det();
        if (det > -0.0000000001 && det < 0.0000000001) {
            cout << "Warning: invert ("<< this->_rows << "x" << this->_cols;
            cout << ") --> matrix not invertible, det: "<< det <<"\n";
            return false;
        }
        det = 1.0/det;

        //Direct inversion method for 1x1, 2x2 and 3x3 matrices.
        if (this->_rows == 1) {
            this->_data[0] = det;
            return true;
        }

        if (this->_rows == 2) {
            //Do the invert.
            temp = this->_data[0];
            this->_data[0] = this->_data[3] * det;
            this->_data[1] = -this->_data[1] * det;
            this->_data[2] = -this->_data[2] * det;
            this->_data[3] = temp * det;
            return true;
        }

        if (this->_rows == 3) {
            temp = this->_data[0];
            this->transpose();

            //Getting a copy of our matrix, it is necessary to compute the result.
            matrix copy_m(*this);
            float * copy_d = copy_m.data();
            //First column.
            this->_data[0] = (copy_d[4] * copy_d[8] - copy_d[5] * copy_d[7]) * det;
            this->_data[1] = (copy_d[3] * copy_d[8] - copy_d[5] * copy_d[6]) * -det;
            this->_data[2] = (copy_d[3] * copy_d[7] - copy_d[4] * copy_d[6]) * det;

            //Second column.
            this->_data[3] = (copy_d[1] * copy_d[8] - copy_d[2] * copy_d[7]) * -det;
            this->_data[4] = (copy_d[0] * copy_d[8] - copy_d[2] * copy_d[6]) * det;
            this->_data[5] = (copy_d[0] * copy_d[7] - copy_d[1] * copy_d[6]) * -det;

            //Third column.
            this->_data[6] = (copy_d[1] * copy_d[5] - copy_d[2] * copy_d[4]) * det;
            this->_data[7] = (copy_d[0] * copy_d[5] - copy_d[2] * copy_d[3]) * -det;
            this->_data[8] = (copy_d[0] * copy_d[4] - copy_d[1] * copy_d[3]) * det;
            return true;
        }

    } else { //else if (this->_rows < 4)

        //Iterative inversion method for matrices bigger than 3x3.
        matrix p(this->_rows, this->_rows);
        p.set_identity();

        //Computing inverse matrix using Gauss reduction.
        for(unsigned i=0; i<p.rows(); i++) {
            temp = this->_data[i*this->_cols + i];
            if (temp > -0.00001 && temp < 0.00001) {
                cout << "Warning: invert ("<< this->_rows << "x" << this->_cols;
                cout << ") --> matrix not invertible\n";
                //cout << "TRACE:\n" << p.debug() << endl;
                return false;
            }
            //Dividing row by the temp value.
            if (temp < 0.99999 || temp > 1.00001) {
                temp = 1.0/temp;
                for(unsigned j=i; j<p.cols(); j++) {
                    this->_data[i*this->_cols + j] *= temp;
                }
                for(unsigned j=0; j<p.cols(); j++) {
                    p[i][j] *= temp;
                }
            }

            //Subtracting actual row to the rest of rows in order to anulate them.
            for(unsigned k=0; k<p.rows(); k++) {
                if (k != i) {
                    temp = this->_data[k*this->_cols + i];
                    if (temp < 0.99999 || temp > 1.00001) {
                        for(unsigned j=0; j<p.cols(); j++) {
                            p[k][j] -= temp * p[i][j];
                            this->_data[k*this->_cols + j] -= temp * this->_data[i*this->_cols + j];
                        }
                    }
                }
            }
        }

        this->set(p);
        return true;

    } //else if (this->_rows < 4)
    return true;
}



float matrix::det() {
    if (this->_rows != this->_cols) {
        cout << "Error: det() --> Non-squared matrix\n";
        return 0.0;
    }

    if (this->_rows == 0 || this->_cols == 0) {
        cout << "Error: det() --> Empty matrix\n";
        return 0.0;
    }

    if (this->_rows == 1) {
        return this->_data[0];
    }

    if (this->_rows == 2) {
        return this->_data[0] * this->_data[3] - this->_data[1] * this->_data[2];
    }

    if (this->_rows == 3) {
        float result = this->_data[0] * this->_data[4] * this->_data[8];
        result += this->_data[1] * this->_data[5] * this->_data[6];
        result += this->_data[2] * this->_data[3] * this->_data[7];

        result -= this->_data[2] * this->_data[4] * this->_data[6];
        result -= this->_data[1] * this->_data[3] * this->_data[8];
        result -= this->_data[0] * this->_data[5] * this->_data[7];
        return result;
    }

    cout << "Error: det() --> not implemented with matrices bigger than 3x3\n";
    return 0.0;
}



bool matrix::ortonormalize() {
    if (this->_rows != 3 || this->_cols != 3) {
        return false;
    }
    //Normalizing first vector.
    vec3 * v1 = (vec3 *) this->_data;
    v1->normalize();

    vec3 * v2 = (vec3 *) &(this->_data[3]);
    vec3 temp;
    mul(&temp, dot(v1, v2), v1);
    sub(v2, v2, &temp);
    v2->normalize();

    vec3 * v3 = (vec3 *) &(this->_data[6]);
    cross(v3, v1, v2);

    cout << "Testing: \n";
    cout << "Dot products:\n";
    cout << "   v1*v2 = " << dot(v1, v2) << endl;
    cout << "   v1*v3 = " << dot(v1, v3) << endl;
    cout << "   v2*v3 = " << dot(v2, v3) << endl << endl;
    cout << "Modules:\n";
    cout << "   v1 = "<< v1->length() <<"\n";
    cout << "   v2 = "<< v2->length() <<"\n";
    cout << "   v3 = "<< v3->length() <<"\n";
    return true;
}


bool matrix::is_symetric() {
    for (unsigned i=0; i<this->_rows; i++) {
        for (unsigned j=0; j<i; j++) {
            if (this->_data[i * this->_cols + j] != this->_data[j * this->_cols + i]) {
                return false;
            }
        }
    }
    return true;
}



bool matrix::factorize() {
    if (this->_rows != this->_cols || this->_rows == 0 || this->_cols == 0) {
        cout << "Error: matrix::factorize --> Matrix not squared or empty (";
        cout << this->_rows <<"x"<< this->_cols <<")"<< endl;
        return false;
    }
    if (!this->is_symetric()) {
        cout << "Error: matrix::factorize --> Matrix not symetric (";
        cout << this->_rows <<"x"<< this->_cols <<")"<< endl;
        return false;
    }

    //Factorizing using Choleski:
    static matrix L;
    L.resize(this->_rows, this->_cols);
    L.set(0.0);
    float pivot = 1.0;
    float summatory = 0.0;

    for (unsigned j=0; j<this->_rows; j++) {

        pivot = this->_data[j * this->_cols + j];
        summatory = 0.0;
        for (unsigned k=0; k<j; k++) {
            summatory += L[j][k] * L[j][k];
        }
        pivot -= summatory;

        if (pivot < 0.000001) {
            cout << "Error: matrix::factorize --> Negative value at (";
            cout << j <<","<< j <<") --> " << pivot << endl;
            cout << "debug:\n" << L.to_s() << endl;
            return false;
        }

        L[j][j] = sqrt(pivot);
        pivot = 1.0 / L[j][j];


        for (unsigned i=j+1; i<this->_rows; i++) {
            summatory = 0.0;
            for (unsigned k=0; k<j; k++) {
                summatory += L[i][k] * L[j][k];
            }
            L[i][j] = pivot * (this->_data[i * this->_cols + j] - summatory);
        }
    }
    this->set(L);
    return true;
}


void matrix::negate() {
    float * p = (float *) this->_data;
    for (unsigned i=0; i<this->_rows * this->_cols; i++) {
        *p = -*p;
        p++;
    }
}




void matrix::operator += (matrix & other) {
    if (this->_rows == other.rows() && this->_cols == other.cols()) {
        for (unsigned i=0; i<this->_cols * this->_rows; i++) {
            this->_data[i] += other.data()[i];
        }
    } else {
        cout << "Error: matrix::operator +=  Cannot sum a matrix of ";
        cout << this->_rows << "x" << this->_cols << " with another of ";
        cout << other.rows() << "x" << other.cols() << endl;
    }
}



void matrix::operator -= (matrix & other) {
    if (this->_rows == other.rows() && this->_cols == other.cols()) {
        for (unsigned i=0; i<this->_cols * this->_rows; i++) {
            this->_data[i] -= other.data()[i];
        }
    } else {
        cout << "Error: matrix::operator -=  Cannot subtract a matrix of ";
        cout << this->_rows << "x" << this->_cols << " with another of ";
        cout << other.rows() << "x" << other.cols() << endl;
    }
}



//~ void matrix::operator *= (matrix & other) {
    //~ if (this->_cols == other.rows()) {
//~ //TODO: ME QUEDE AQUI.
    //~ } else {
        //~ cout << "Error: matrix::operator *=  Cannot multiply a matrix of ";
        //~ cout << this->_rows << "x" << this->_cols << " by another of ";
        //~ cout << other.rows() << "x" << other.cols() << endl;
    //~ }
//~ }


bool matrix::operator == (const matrix & other) const {
    if (this->_rows != other.rows() || this->_cols != other.cols()) {
        return false;
    }
    float * p = (float *) this->_data;
    float * q = (float *) other.data();
    for (unsigned i=0; i<this->_rows*this->_cols; i++) {
        if (*p != *q) {
            return false;
        }
        p++;
        q++;
    }
    return true;
}



bool matrix::operator != (const matrix & other) const {
    return !(*this == other);
}




bool matrix::is_defined_diag() const {
    unsigned pi = min(this->_rows, this->_cols);
    for (unsigned i=0; i<pi; i++) {
        pi = i*this->_cols + i;
        if (this->_data[pi] < 0.00001 && this->_data[pi] > -0.00001) {
            return false;
        }
    }
    return true;
}


float matrix::summatory() const {
    float result = 0.0;
    float * p = this->_data;
    for (unsigned i=0; i<this->_rows * this->_cols; i++) {
        result += (*p);
        p++;
    }
    return result;
}


float matrix::abs_summatory() const {
    float result = 0.0;
    float * p = this->_data;
    for (unsigned i=0; i<this->_rows * this->_cols; i++) {
        result += fabs(*p);
        p++;
    }
    return result;
}


void matrix::set_submatrix(matrix & m) {
    unsigned min_rows = min(this->_rows, m.rows());
    unsigned min_cols = min(this->_cols, m.cols());
    for (unsigned i=0; i<min_rows; i++) {
        for (unsigned j=0; j<min_cols; j++) {
            this->_data[this->_rows * i + j] = m[i][j];
        }
    }
}



void matrix::set_submatrix(unsigned rows, unsigned cols, matrix & m) {
    if (rows <= m.rows() && cols <= m.cols()) {
        for (unsigned i=0; i<rows; i++) {
            for (unsigned j=0; j<cols; j++) {
                this->_data[this->_rows * i + j] = m[i][j];
            }
        }
    }
}



void matrix::map(unsigned rows, unsigned cols, float * external) {
    if (external == NULL) {
        cout << "Error: matrix::map --> external data is NULL\n";
        return;
    }

    if (this->_data == this->_local.data()) {
        this->_prev_rows = this->_rows;
        this->_prev_cols = this->_cols;
    }

    this->_rows = rows;
    this->_cols = cols;
    this->_data = external;
}



void matrix::unmap() {
    if (this->_data != this->_local.data()) {
        this->_data = (float *) this->_local.data();
        this->_rows = this->_prev_rows;
        this->_cols = this->_prev_cols;
    } else {
        cout << "matrix::unmap --> Doing nothing\n";
    }
}



bool add(matrix & result, const matrix & op1, const matrix & op2) {
    if (op1.rows() != op2.rows() || op1.cols() != op2.cols()) {
        cout << "Error: add --> op1 ("<< op1.rows() <<"x"<< op1.cols();
        cout <<") and op2 ("<< op2.rows() <<"x"<< op2.rows() <<")";
        cout << " cannot be added: different row or column number\n";
        return false;
    }
    if (result.rows() != op1.rows() || result.cols() != op2.cols()) {
        result.resize(op1.rows(), op1.cols());
    }
    //Performing the Sum.
    float * a = (float *) op1.data();
    float * b = (float *) op2.data();
    float * c = (float *) result.data();

    for (unsigned i=0; i < op1.rows() * op1.cols(); i++) {
        *c++ = *a++ + *b++;
    }
    return true;
}



bool mul(matrix & result, const matrix & op1, const matrix & op2) {

    if (op1.rows() == 1 && op1.cols() == 1) {
        return mul(result, op1[0][0], op2);
    }

    if (op1.cols() != op2.rows()) {
        cout << "Error: mul --> op1 and op2 have different row or column number ";
        cout << "["<< op1.rows() << ", " << op1.cols() <<"] vs ["<< op2.rows() << ", " << op2.cols() <<"]\n";
        return false;
    }
    float temp;
    if (result.rows() != op1.rows() || result.cols() != op2.cols()) {
        result.resize(op1.rows(), op2.cols());
    }
    //Performing the multiplication.
    for (unsigned i=0; i<op1.rows(); i++) {
        for (unsigned j=0; j<op2.cols(); j++) {
            temp = 0.0;
            for (unsigned k=0; k<op1.cols(); k++) {
                temp += op1[i][k] * op2[k][j];
            }
            result[i][j] = temp;
        }
    }
    return true;
}




bool accum(matrix & result, const matrix & op1, const matrix & op2) {
    if (op1.rows() != op2.rows() || op1.cols() != op2.cols()) {
        cout << "Error: accum --> op1 ("<< op1.rows() <<"x"<< op1.cols();
        cout <<") and op2 ("<< op2.rows() <<"x"<< op2.rows() <<")";
        cout << " cannot be added: different row or column number\n";
        return false;
    }
    if (result.rows() != op1.rows() || result.cols() != op2.cols()) {
        result.resize(op1.rows(), op1.cols());
    }
    //Performing the Sum.
    float * a = (float *) op1.data();
    float * b = (float *) op2.data();
    float * c = (float *) result.data();

    for (unsigned i=0; i < op1.rows() * op1.cols(); i++) {
        *c++ += *a++ + *b++;
    }
    return true;
}



bool mul(matrix & result, const float v, const matrix & op) {
    if (result.rows() != op.rows() || result.cols() != op.cols()) {
        result.resize(op.rows(), op.cols());
    }
    //Performing the Multiplication.
    float * a = (float *) op.data();
    float * r = (float *) result.data();

    for (unsigned i=0; i < op.rows() * op.cols(); i++) {
        *r++ = *a++ * v;
    }
    return true;
}



bool mul(vec4 & result, const matrix & op1, const vec4 & op2) {
    if (op1.cols() == 4) {
        for (unsigned i=0; i<4; i++) {
            result[i] = op1[i][0] * op2[0] + op1[i][1] * op2[1] + op1[i][2] * op2[2] + op1[i][3] * op2[3];
        }
    }
    return true;
}


bool sub(matrix & result, const matrix & op1, const matrix & op2) {
    if (op1.rows() != op2.rows() || op1.cols() != op2.cols()) {
        cout << "Error: sub --> op1 ("<< op1.rows() <<"x"<< op1.cols();
        cout <<") and op2 ("<< op2.rows() <<"x"<< op2.rows() <<")";
        cout << " cannot be substracted: different row or column number\n";
        return false;
    }
    if (result.rows() != op1.rows() || result.cols() != op2.cols()) {
        result.resize(op1.rows(), op1.cols());
    }
    //Performing the Sum.
    float * a = (float *) op1.data();
    float * b = (float *) op2.data();
    float * c = result.data();

    for (unsigned i=0; i < op1.rows() * op1.cols(); i++) {
        *c++ = *a++ - *b++;
    }
    return true;
}




bool block_add(unsigned block_rows, unsigned block_cols,
               matrix & dest, unsigned dest_offset_row, unsigned dest_offset_col,
               const matrix & orig, unsigned orig_offset_row, unsigned orig_offset_col) {

    if (orig.rows() < orig_offset_row + block_rows ||
        orig.cols() < orig_offset_col + block_cols ||
        dest.rows() < dest_offset_row + block_rows ||
        dest.cols() < dest_offset_col + block_cols) {
        cout << "ERROR: block_add --> The matrices need to be bigger given the offsets.\n";
        cout << "       block size (" << block_rows << "x" << block_cols << ")\n";
        cout << "       orig ("<< orig.rows() << "x"<< orig.cols();
        cout << ") offset_row: " << orig_offset_row << "  offset_col: "<< orig_offset_col << endl;
        cout << "       dest ("<< dest.rows() << "x"<< dest.cols();
        cout << ") offset_row: " << dest_offset_row << "  offset_col: "<< dest_offset_col << endl;
        return false;
    }

    for (unsigned i=0; i<block_rows; i++) {
        for (unsigned j=0; j<block_cols; j++) {
            dest[dest_offset_row+i][dest_offset_col+j] += orig[orig_offset_row+i][orig_offset_col+j];
        }
    }
    return true;
}



bool block_add(matrix & dest, unsigned dest_offset_row, unsigned dest_offset_col,
               const matrix & orig) {

    return block_add(orig.rows(), orig.cols(),
                     dest, dest_offset_row, dest_offset_col,
                     orig, 0, 0);
}



bool block_set(unsigned block_rows, unsigned block_cols,
               matrix & dest, unsigned dest_offset_row, unsigned dest_offset_col,
               const matrix & orig, unsigned orig_offset_row, unsigned orig_offset_col) {

    if (orig.rows() < orig_offset_row + block_rows ||
        orig.cols() < orig_offset_col + block_cols ||
        dest.rows() < dest_offset_row + block_rows ||
        dest.cols() < dest_offset_col + block_cols) {
        cout << "ERROR: block_set --> The matrices need to be bigger given the offsets.\n";
        cout << "       block size (" << block_rows << "x" << block_cols << ")\n";
        cout << "       orig ("<< orig.rows() << "x"<< orig.cols();
        cout << ") offset_row: " << orig_offset_row << "  offset_col: "<< orig_offset_col << endl;
        cout << "       dest ("<< dest.rows() << "x"<< dest.cols();
        cout << ") offset_row: " << dest_offset_row << "  offset_col: "<< dest_offset_col << endl;
        return false;
    }

    for (unsigned i=0; i<block_rows; i++) {
        for (unsigned j=0; j<block_cols; j++) {
            dest[dest_offset_row+i][dest_offset_col+j] = orig[orig_offset_row+i][orig_offset_col+j];
        }
    }
    return true;
}




bool block_set(matrix & dest, unsigned dest_offset_row, unsigned dest_offset_col,
               const matrix & orig) {
    return block_set(orig.rows(), orig.cols(),
                     dest, dest_offset_row, dest_offset_col,
                     orig, 0, 0);
}



void apply_preconditioner(matrix & A, matrix & b) {
    matrix c(A.rows(), A.cols());
    //Applying preconditioner.
    //c.resize(A.rows(), A.cols());
    c.set(0.0);
    for (unsigned i=0; i<A.cols(); i++) {
        if (A[i][i] < -0.000001 || A[i][i] > 0.000001) {
            c[i][i] = 1.0/A[i][i];
        }
    }
    matrix temp;
    mul(temp, c, A);
    A.set(temp);

    mul(temp, c, b);
    b.set(temp);
}


///Solve a linear system of ecuations, using Gauss-Seidel method.
///http://es.wikipedia.org/wiki/M%C3%A9todo_de_Gauss-Seidel

///WARNING: This method only works when the diagonal of A is bigger than the
///         rest of values. Otherwise never converges.

bool solve_gs(const matrix & A, matrix & x, const matrix & b) {

    if (b.cols() != 1 || A.rows() != b.rows() || A.rows() != A.cols()) {
        cout << "Error: solve --> A is "<< A.rows() <<"x"<< A.cols();
        cout <<" and b is "<< b.rows() <<"x"<< b.cols() << " INCORRECT\n";
        return false;
    }

    if (!A.is_defined_diag()) {
        cout << "Error: solve --> A has zeroes in the main diag.\n";
        return false;
    }

    if (x.cols() != 1 || A.cols() != x.rows()) {
        x.resize(A.cols(), 1);
    }

    //Initial approximation.
    x.set(1.0);
    //Matrix to compute if the solution is near the convergence.
    matrix prev_x(x);

    matrix distance;
    unsigned iterations = 0;

    //Starting Gauss-Seidel Method.
    bool convergence = false;
    float temp;

    while (!convergence) {
        //Gauss-Seidel.
        for (unsigned i=0; i<A.rows(); i++) {
            temp = 0.0;
            for (unsigned j=0; j<A.cols(); j++) {
                if (j != i) {
                    temp += A[i][j] * x[j][0];
                }
            }
            x[i][0] = (b[i][0] - temp) / A[i][i];
        }

        //Testing convergence.
        iterations++;
        sub(distance, x, prev_x);

        prev_x.set(x);
        if (distance.abs_summatory() < 0.0001) {
            convergence = true;
        }
    }
    cout << "Iterations " << iterations << endl;
    return true;
}



///Solve the linear system Ax = b using Conjugate Gradient
bool solve_gc(const matrix & A, matrix & x, const matrix & b) {

    if (b.cols() != 1 || A.rows() != b.rows() || A.rows() != A.cols()) {
        cout << "Error: solve --> A is "<< A.rows() <<"x"<< A.cols();
        cout <<" and b is "<< b.rows() <<"x"<< b.cols() << " INCORRECT\n";
        return false;
    }
    //Initial approximation of x.
    x.set(0.0);

    matrix c, r, v, alpha, beta, temp1, temp2, temp3, temp4, temp5, temp6, v_t, Av, v_tAv;
    //Computing the residue.
    //mul(temp1, A, x);
    //sub(r, b, temp1);
    //v.set(r);
    r.set(b);
    v.set(b);

    unsigned iterations = 0;

    //When the residue is small enough, the (approximate) solution is found.
    while (iterations < 60 && r.abs_summatory() > 0.00001) {
        mul(Av, A, v);
        //Computing Alpha
        v_t.set(v);
        v_t.transpose();

        mul(v_tAv, v_t, Av);
        if (!v_tAv.invert()) {
            iterations++;
            continue;
        }

        mul(temp1, v_t, r);
        mul(alpha, temp1, v_tAv);

        //Computing next X.
        mul(temp2, alpha, v);
        add(x, x, temp2);

        //Computing next residue.
        mul(temp3, alpha, Av);
        sub(r, r, temp3);

        //Computing Beta.
        mul(temp4, v_t, A);
        mul(temp5, temp4, r);
        mul(beta, temp5, v_tAv);
        beta.negate();

        //Computing next v.
        mul(temp6, beta, v);
        add(v, r, temp6);
        iterations++;
    }
    cout << "Iterations " << iterations << endl;
    return true;
}


bool solve_choleski(const matrix & A, matrix & x, const matrix & b) {
    cout << "\n\nERROR: solve_choleski --> Not implemented yet!!\n\n";
    return false;
}


bool solve_choleski(const matrix & A, const matrix & A_t, matrix & x, const matrix & b) {
    cout << "\n\nERROR: solve_choleski --> Not implemented yet!!\n\n";
    return false;
}


bool solve_choleski(const matrix & A, const matrix & d, const matrix & A_t, matrix & x, const matrix & b) {
    cout << "\n\nERROR: solve_choleski --> Not implemented yet!!\n\n";
    return false;
}


matrix create_rotation_matrix_around_X(float degrees) {
    matrix R(3,3);
    //    1    0      0
    //    0   cosA  -sinA
    //    0   sinA   cosA
    float radians = degrees * DEG_TO_RAD;
    R[0][0] = 1.0;
    R[0][1] = R[0][2] = R[1][0] = R[2][0] = 0.0;
    R[1][1] = R[2][2] = cos(radians);
    float sinA = sin(radians);
    R[1][2] = -sinA;
    R[2][1] = sinA;
    return R;
}


matrix create_rotation_matrix_around_Y(float degrees) {
    matrix R(3,3);
    //    cosA   0   sinA
    //      0    1    0
    //    -sinA   0   cosA
    float radians = degrees * DEG_TO_RAD;
    R[0][0] = R[2][2] = cos(radians);
    R[1][1] = 1.0;
    R[0][1] = R[1][0] = R[1][2] = R[2][1] = 0.0;
    float sinA = sin(radians);
    R[0][2] = sinA;
    R[2][0] = -sinA;
    return R;
}


matrix create_rotation_matrix_around_Z(float degrees) {
    matrix R(3,3);
    //    cosA -sinA  0
    //    sinA  cosA  0
    //      0     0   1
    float radians = degrees * DEG_TO_RAD;
    R[0][0] = R[1][1] = cos(radians);
    float sinA = sin(radians);
    R[0][1] = -sinA;
    R[1][0] = sinA;
    R[2][2] = 1.0;
    R[0][2] = R[1][2] = R[2][0] = R[2][1] = 0.0;
    return R;
}




fmatrix::fmatrix() {
    this->_data = (float *) this->_local.data();
    this->clear();
}


fmatrix::fmatrix(unsigned rows, unsigned cols) {
    this->_local.resize(rows * cols);
    this->_data = (float *) this->_local.data();
    this->_rows = rows;
    this->_cols = cols;
}



fmatrix::fmatrix(unsigned rows, unsigned cols, const vector<float> & data) {
    this->_rows = rows;
    this->_cols = cols;

    if (rows == 0 || cols == 0) {
        cout << "Error: fmatrix::fmatrix(), a fmatrix cannot have zero rows or zero cols\n";
        return;
    }

    this->resize(rows, cols, data);
    this->_data = (float *) this->_local.data();
}



fmatrix::fmatrix(unsigned rows, unsigned cols, float * data) {
    this->_data = (float *) this->_local.data();
    this->_rows = rows;
    this->_cols = cols;

    if (rows == 0 || cols == 0) {
        cout << "Error: fmatrix::fmatrix(), a fmatrix cannot have zero rows or zero cols\n";
        return;
    }
    if (data == NULL) {
        cout << "Error: fmatrix::fmatrix(), data is null\n";
        return;
    }
    this->resize(rows, cols, data);
}



fmatrix::fmatrix(float x, float y, float z) {
    this->_data = (float *) this->_local.data();
    this->_rows = 1;
    this->_cols = 3;
    this->_local.resize(3);
    this->_local[0] = x;
    this->_local[1] = y;
    this->_local[2] = z;
}


fmatrix::fmatrix(const fmatrix & original) {
    this->_data = (float *) this->_local.data();
    this->_rows = original.rows();
    this->_cols = original.cols();

    if (_rows == 0 || _cols == 0) {
        cout << "Error: fmatrix::fmatrix(), a fmatrix cannot have zero rows or zero cols\n";
        return;
    }
    this->resize(original.rows(), original.cols(), original.data());
}


fmatrix::~fmatrix() {
    this->clear();
}


void fmatrix::clear() {
    this->_rows = 0;
    this->_cols = 0;

    if (this->_data != this->_local.data()) {
        this->unmap();
    }
    this->_local.clear();
}


unsigned fmatrix::rows() const {
    return this->_rows;
}


unsigned fmatrix::cols() const {
    return this->_cols;
}


float * fmatrix::data() {
    return this->_data;
}


const float * fmatrix::data() const {
    return this->_data;
}


//~ const vector<float> & fmatrix::data() const {
    //~ return *this->_data;
//~ }


void fmatrix::set_identity() {
    for (unsigned i=0; i<this->_rows; i++) {
        for (unsigned j=0; j<this->_cols; j++) {
            if (i == j) {
                this->_data[i*this->_rows + j] = (float) 1.0;
            } else {
                this->_data[i*this->_rows + j] = (float) 0.0;
            }
        }
    }
}



bool fmatrix::is_identity() {
    if (this->_rows != this->_cols) {
        return false;
    }
    float * p = (float *) this->_data;
    for (unsigned i=0; i<this->_rows; i++) {
        for (unsigned j=0; j<this->_cols; j++) {
            if (i==j) {
                if (*p < 0.99999 || *p > 1.00001) {
                    return false;
                }
            } else {
                if (*p < -0.00001 || *p > 0.00001) {
                    return false;
                }
            }
            p++;
        }
    }
    return true;
}




bool fmatrix::is_zero() {
    for (unsigned i=0; i<this->_rows * this->_cols; i++) {
        if (this->_data[i] < -0.00001 || this->_data[i] > 0.00001) {
            return false;
        }
    }
    return true;
}



void fmatrix::set(float v) {
    for (unsigned i=0; i<this->_rows * this->_cols; i++) {
        this->_data[i] = v;
    }
}



void fmatrix::set(const fmatrix & v) {
    if (this->_rows != v.rows() || this->_cols != v.cols()) {
        this->resize(v.rows(), v.cols(), v.data());
    } else {
        for (unsigned i=0; i<this->_rows * this->_cols; i++) {
            this->_data[i] = v.data()[i];
        }
    }
}


void fmatrix::clone(fmatrix & v) {
    if (this->_rows != v.rows() || this->_cols != v.cols()) {
        v.resize(this->_rows, this->_cols, this->_data);
    } else {
        for (unsigned i=0; i<this->_rows * this->_cols; i++) {
            v.data()[i] = this->_data[i];
        }
    }
}


void fmatrix::resize(unsigned rows, unsigned cols) {
    if (rows != this->_rows || cols != this->_cols) {
        this->_rows = rows;
        this->_cols = cols;
        if (this->_data == this->_local.data()) {
            //~ cout << "resizing\n";
            this->_local.resize(rows * cols);
            this->_data = (float *) this->_local.data();
        } else {
            cout << "fmatrix::resize --> Cannot resize mapped data\n";
        }
    }
}


void fmatrix::resize(unsigned rows, unsigned cols, const vector<float> & data) {
    this->_rows = rows;
    this->_cols = cols;
    if (this->_data == this->_local.data()) {
        this->_local.assign(data.begin(), data.end());
        this->_data = (float *) this->_local.data();
    } else {
        cout << "fmatrix::resize --> Cannot resize mapped data\n";
    }
}


void fmatrix::resize(unsigned rows, unsigned cols, const float * data) {
    this->_rows = rows;
    this->_cols = cols;
    if (this->_data == this->_local.data()) {
        this->_local.resize(rows * cols);
        this->_data = (float *) this->_local.data();
        for (unsigned i=0; i<rows*cols; i++) {
            this->_local[i] = data[i];
        }
    } else {
        cout << "fmatrix::resize --> Cannot resize mapped data\n";
    }
}


string fmatrix::to_s() {
    if (this->_rows == 0 || this->_cols == 0) {
        return "Empty fmatrix\n";
    }
    string result = "";

    //cout << "Debug: Real length: " << this->_length << " used length: " << this->_rows * this->_cols << endl;

    for (unsigned i=0; i<this->_rows; i++) {
        for (unsigned j=0; j<this->_cols; j++) {
            if (this->_data[i*this->_cols + j] < 0.00001) {
                result += "  " + float_to_str(this->_data[i*this->_cols + j]);
            } else {
                result += "   " + float_to_str(this->_data[i*this->_cols + j]);
            }
        }
        result += "\n";
    }
    return result;
}


void fmatrix::transpose() {
    if (this->_rows == 0 || this->_cols == 0) {
        cout << "Error: Transpose --> Cannot transpose an empty fmatrix.\n";
        return;
    }

    if (this->_cols == 1 || this->_rows == 1) {
        unsigned temp = this->_rows;
        this->_rows = this->_cols;
        this->_cols = temp;
        return;
    }

    vector<float> aux;
    aux.resize(this->_rows * this->_cols, 0.0);

    for (unsigned i=0; i<this->_rows; i++) {
        for (unsigned j=0; j<this->_cols; j++) {
            aux[j * this->_rows + i] = this->_data[i * this->_cols + j];
        }
    }
    unsigned temp = this->_rows;
    this->_rows = this->_cols;
    this->_cols = temp;

    for (unsigned i=0; i<this->_rows*this->_cols; i++) {
        this->_data[i] = aux.data()[i];
    }

    aux.clear();
}



bool fmatrix::invert() {

    if (this->_rows != this->_cols) {
        cerr << "ERROR: fmatrix::invert --> Non square fmatrix\n";
        return false;
    }

    float temp;

    if (this->_rows < 4) {

        float det = this->det();
        if (det > -0.0000000001 && det < 0.0000000001) {
            cout << "Warning: invert ("<< this->_rows << "x" << this->_cols;
            cout << ") --> fmatrix not invertible, det: "<< det <<"\n";
            return false;
        }
        det = 1.0/det;

        //Direct inversion method for 1x1, 2x2 and 3x3 matrices.
        if (this->_rows == 1) {
            this->_data[0] = det;
            return true;
        }

        if (this->_rows == 2) {
            //Do the invert.
            temp = this->_data[0];
            this->_data[0] = this->_data[3] * det;
            this->_data[1] = -this->_data[1] * det;
            this->_data[2] = -this->_data[2] * det;
            this->_data[3] = temp * det;
            return true;
        }

        if (this->_rows == 3) {
            temp = this->_data[0];
            this->transpose();

            //Getting a copy of our fmatrix, it is necessary to compute the result.
            fmatrix copy_m(*this);
            float * copy_d = copy_m.data();
            //First column.
            this->_data[0] = (copy_d[4] * copy_d[8] - copy_d[5] * copy_d[7]) * det;
            this->_data[1] = (copy_d[3] * copy_d[8] - copy_d[5] * copy_d[6]) * -det;
            this->_data[2] = (copy_d[3] * copy_d[7] - copy_d[4] * copy_d[6]) * det;

            //Second column.
            this->_data[3] = (copy_d[1] * copy_d[8] - copy_d[2] * copy_d[7]) * -det;
            this->_data[4] = (copy_d[0] * copy_d[8] - copy_d[2] * copy_d[6]) * det;
            this->_data[5] = (copy_d[0] * copy_d[7] - copy_d[1] * copy_d[6]) * -det;

            //Third column.
            this->_data[6] = (copy_d[1] * copy_d[5] - copy_d[2] * copy_d[4]) * det;
            this->_data[7] = (copy_d[0] * copy_d[5] - copy_d[2] * copy_d[3]) * -det;
            this->_data[8] = (copy_d[0] * copy_d[4] - copy_d[1] * copy_d[3]) * det;
            return true;
        }

    } else { //else if (this->_rows < 4)

        //Iterative inversion method for matrices bigger than 3x3.
        fmatrix p(this->_rows, this->_rows);
        p.set_identity();

        //Computing inverse fmatrix using Gauss reduction.
        for(unsigned i=0; i<p.rows(); i++) {
            temp = this->_data[i*this->_cols + i];
            if (temp > -0.00001 && temp < 0.00001) {
                cout << "Warning: invert ("<< this->_rows << "x" << this->_cols;
                cout << ") --> fmatrix not invertible\n";
                //cout << "TRACE:\n" << p.debug() << endl;
                return false;
            }
            //Dividing row by the temp value.
            if (temp < 0.99999 || temp > 1.00001) {
                temp = 1.0/temp;
                for(unsigned j=i; j<p.cols(); j++) {
                    this->_data[i*this->_cols + j] *= temp;
                }
                for(unsigned j=0; j<p.cols(); j++) {
                    p[i][j] *= temp;
                }
            }

            //Subtracting actual row to the rest of rows in order to anulate them.
            for(unsigned k=0; k<p.rows(); k++) {
                if (k != i) {
                    temp = this->_data[k*this->_cols + i];
                    if (temp < 0.99999 || temp > 1.00001) {
                        for(unsigned j=0; j<p.cols(); j++) {
                            p[k][j] -= temp * p[i][j];
                            this->_data[k*this->_cols + j] -= temp * this->_data[i*this->_cols + j];
                        }
                    }
                }
            }
        }

        this->set(p);
        return true;

    } //else if (this->_rows < 4)
    return true;
}



float fmatrix::det() {
    if (this->_rows != this->_cols) {
        cout << "Error: det() --> Non-squared fmatrix\n";
        return 0.0;
    }

    if (this->_rows == 0 || this->_cols == 0) {
        cout << "Error: det() --> Empty fmatrix\n";
        return 0.0;
    }

    if (this->_rows == 1) {
        return this->_data[0];
    }

    if (this->_rows == 2) {
        return this->_data[0] * this->_data[3] - this->_data[1] * this->_data[2];
    }

    if (this->_rows == 3) {
        float result = this->_data[0] * this->_data[4] * this->_data[8];
        result += this->_data[1] * this->_data[5] * this->_data[6];
        result += this->_data[2] * this->_data[3] * this->_data[7];

        result -= this->_data[2] * this->_data[4] * this->_data[6];
        result -= this->_data[1] * this->_data[3] * this->_data[8];
        result -= this->_data[0] * this->_data[5] * this->_data[7];
        return result;
    }

    cout << "Error: det() --> not implemented with matrices bigger than 3x3\n";
    return 0.0;
}



bool fmatrix::ortonormalize() {
    if (this->_rows != 3 || this->_cols != 3) {
        return false;
    }
    //Normalizing first vector.
    vec3 * v1 = (vec3 *) this->_data;
    v1->normalize();

    vec3 * v2 = (vec3 *) &(this->_data[3]);
    vec3 temp;
    mul(&temp, dot(v1, v2), v1);
    sub(v2, v2, &temp);
    v2->normalize();

    vec3 * v3 = (vec3 *) &(this->_data[6]);
    cross(v3, v1, v2);

    cout << "Testing: \n";
    cout << "Dot products:\n";
    cout << "   v1*v2 = " << dot(v1, v2) << endl;
    cout << "   v1*v3 = " << dot(v1, v3) << endl;
    cout << "   v2*v3 = " << dot(v2, v3) << endl << endl;
    cout << "Modules:\n";
    cout << "   v1 = "<< v1->length() <<"\n";
    cout << "   v2 = "<< v2->length() <<"\n";
    cout << "   v3 = "<< v3->length() <<"\n";
    return true;
}


bool fmatrix::is_symetric() {
    for (unsigned i=0; i<this->_rows; i++) {
        for (unsigned j=0; j<i; j++) {
            if (this->_data[i * this->_cols + j] != this->_data[j * this->_cols + i]) {
                return false;
            }
        }
    }
    return true;
}



bool fmatrix::factorize() {
    if (this->_rows != this->_cols || this->_rows == 0 || this->_cols == 0) {
        cout << "Error: fmatrix::factorize --> Matrix not squared or empty (";
        cout << this->_rows <<"x"<< this->_cols <<")"<< endl;
        return false;
    }
    if (!this->is_symetric()) {
        cout << "Error: fmatrix::factorize --> Matrix not symetric (";
        cout << this->_rows <<"x"<< this->_cols <<")"<< endl;
        return false;
    }

    //Factorizing using Choleski:
    static fmatrix L;
    L.resize(this->_rows, this->_cols);
    L.set(0.0);
    float pivot = 1.0;
    float summatory = 0.0;

    for (unsigned j=0; j<this->_rows; j++) {

        pivot = this->_data[j * this->_cols + j];
        summatory = 0.0;
        for (unsigned k=0; k<j; k++) {
            summatory += L[j][k] * L[j][k];
        }
        pivot -= summatory;

        if (pivot < 0.000001) {
            cout << "Error: fmatrix::factorize --> Negative value at (";
            cout << j <<","<< j <<") --> " << pivot << endl;
            cout << "debug:\n" << L.to_s() << endl;
            return false;
        }

        L[j][j] = sqrt(pivot);
        pivot = 1.0 / L[j][j];


        for (unsigned i=j+1; i<this->_rows; i++) {
            summatory = 0.0;
            for (unsigned k=0; k<j; k++) {
                summatory += L[i][k] * L[j][k];
            }
            L[i][j] = pivot * (this->_data[i * this->_cols + j] - summatory);
        }
    }
    this->set(L);
    return true;
}


void fmatrix::negate() {
    float * p = (float *) this->_data;
    for (unsigned i=0; i<this->_rows * this->_cols; i++) {
        *p = -*p;
        p++;
    }
}




void fmatrix::operator += (fmatrix & other) {
    if (this->_rows == other.rows() && this->_cols == other.cols()) {
        for (unsigned i=0; i<this->_cols * this->_rows; i++) {
            this->_data[i] += other.data()[i];
        }
    } else {
        cout << "Error: fmatrix::operator +=  Cannot sum a fmatrix of ";
        cout << this->_rows << "x" << this->_cols << " with another of ";
        cout << other.rows() << "x" << other.cols() << endl;
    }
}



void fmatrix::operator -= (fmatrix & other) {
    if (this->_rows == other.rows() && this->_cols == other.cols()) {
        for (unsigned i=0; i<this->_cols * this->_rows; i++) {
            this->_data[i] -= other.data()[i];
        }
    } else {
        cout << "Error: fmatrix::operator -=  Cannot subtract a fmatrix of ";
        cout << this->_rows << "x" << this->_cols << " with another of ";
        cout << other.rows() << "x" << other.cols() << endl;
    }
}



//~ void fmatrix::operator *= (fmatrix & other) {
    //~ if (this->_cols == other.rows()) {
//~ //TODO: ME QUEDE AQUI.
    //~ } else {
        //~ cout << "Error: fmatrix::operator *=  Cannot multiply a fmatrix of ";
        //~ cout << this->_rows << "x" << this->_cols << " by another of ";
        //~ cout << other.rows() << "x" << other.cols() << endl;
    //~ }
//~ }


bool fmatrix::operator == (const fmatrix & other) const {
    if (this->_rows != other.rows() || this->_cols != other.cols()) {
        return false;
    }
    float * p = (float *) this->_data;
    float * q = (float *) other.data();
    for (unsigned i=0; i<this->_rows*this->_cols; i++) {
        if (*p != *q) {
            return false;
        }
        p++;
        q++;
    }
    return true;
}



bool fmatrix::operator != (const fmatrix & other) const {
    return !(*this == other);
}




bool fmatrix::is_defined_diag() const {
    unsigned pi = min(this->_rows, this->_cols);
    for (unsigned i=0; i<pi; i++) {
        pi = i*this->_cols + i;
        if (this->_data[pi] < 0.00001 && this->_data[pi] > -0.00001) {
            return false;
        }
    }
    return true;
}


float fmatrix::summatory() const {
    float result = 0.0;
    float * p = this->_data;
    for (unsigned i=0; i<this->_rows * this->_cols; i++) {
        result += (*p);
        p++;
    }
    return result;
}


float fmatrix::abs_summatory() const {
    float result = 0.0;
    float * p = this->_data;
    for (unsigned i=0; i<this->_rows * this->_cols; i++) {
        result += fabs(*p);
        p++;
    }
    return result;
}


void fmatrix::set_subfmatrix(fmatrix & m) {
    unsigned min_rows = min(this->_rows, m.rows());
    unsigned min_cols = min(this->_cols, m.cols());
    for (unsigned i=0; i<min_rows; i++) {
        for (unsigned j=0; j<min_cols; j++) {
            this->_data[this->_rows * i + j] = m[i][j];
        }
    }
}



void fmatrix::set_subfmatrix(unsigned rows, unsigned cols, fmatrix & m) {
    if (rows <= m.rows() && cols <= m.cols()) {
        for (unsigned i=0; i<rows; i++) {
            for (unsigned j=0; j<cols; j++) {
                this->_data[this->_rows * i + j] = m[i][j];
            }
        }
    }
}



void fmatrix::map(unsigned rows, unsigned cols, float * external) {
    if (external == NULL) {
        cout << "Error: fmatrix::map --> external data is NULL\n";
        return;
    }

    if (this->_data == this->_local.data()) {
        this->_prev_rows = this->_rows;
        this->_prev_cols = this->_cols;
    }

    this->_rows = rows;
    this->_cols = cols;
    this->_data = external;
}



void fmatrix::unmap() {
    if (this->_data != this->_local.data()) {
        this->_data = (float *) this->_local.data();
        this->_rows = this->_prev_rows;
        this->_cols = this->_prev_cols;
    } else {
        cout << "fmatrix::unmap --> Doing nothing\n";
    }
}



bool add(fmatrix & result, const fmatrix & op1, const fmatrix & op2) {
    if (op1.rows() != op2.rows() || op1.cols() != op2.cols()) {
        cout << "Error: add --> op1 ("<< op1.rows() <<"x"<< op1.cols();
        cout <<") and op2 ("<< op2.rows() <<"x"<< op2.rows() <<")";
        cout << " cannot be added: different row or column number\n";
        return false;
    }
    if (result.rows() != op1.rows() || result.cols() != op2.cols()) {
        result.resize(op1.rows(), op1.cols());
    }
    //Performing the Sum.
    float * a = (float *) op1.data();
    float * b = (float *) op2.data();
    float * c = (float *) result.data();

    for (unsigned i=0; i < op1.rows() * op1.cols(); i++) {
        *c++ = *a++ + *b++;
    }
    return true;
}



bool mul(fmatrix & result, const fmatrix & op1, const fmatrix & op2) {

    if (op1.rows() == 1 && op1.cols() == 1) {
        return mul(result, op1[0][0], op2);
    }

    if (op1.cols() != op2.rows()) {
        cout << "Error: mul --> op1 and op2 have different row or column number ";
        cout << "["<< op1.rows() << ", " << op1.cols() <<"] vs ["<< op2.rows() << ", " << op2.cols() <<"]\n";
        return false;
    }
    float temp;
    if (result.rows() != op1.rows() || result.cols() != op2.cols()) {
        result.resize(op1.rows(), op2.cols());
    }
    //Performing the multiplication.
    for (unsigned i=0; i<op1.rows(); i++) {
        for (unsigned j=0; j<op2.cols(); j++) {
            temp = 0.0;
            for (unsigned k=0; k<op1.cols(); k++) {
                temp += op1[i][k] * op2[k][j];
            }
            result[i][j] = temp;
        }
    }
    return true;
}




bool accum(fmatrix & result, const fmatrix & op1, const fmatrix & op2) {
    if (op1.rows() != op2.rows() || op1.cols() != op2.cols()) {
        cout << "Error: accum --> op1 ("<< op1.rows() <<"x"<< op1.cols();
        cout <<") and op2 ("<< op2.rows() <<"x"<< op2.rows() <<")";
        cout << " cannot be added: different row or column number\n";
        return false;
    }
    if (result.rows() != op1.rows() || result.cols() != op2.cols()) {
        result.resize(op1.rows(), op1.cols());
    }
    //Performing the Sum.
    float * a = (float *) op1.data();
    float * b = (float *) op2.data();
    float * c = (float *) result.data();

    for (unsigned i=0; i < op1.rows() * op1.cols(); i++) {
        *c++ += *a++ + *b++;
    }
    return true;
}



bool mul(fmatrix & result, const float v, const fmatrix & op) {
    if (result.rows() != op.rows() || result.cols() != op.cols()) {
        result.resize(op.rows(), op.cols());
    }
    //Performing the Multiplication.
    float * a = (float *) op.data();
    float * r = (float *) result.data();

    for (unsigned i=0; i < op.rows() * op.cols(); i++) {
        *r++ = *a++ * v;
    }
    return true;
}


void fmul3x3(vec3 & r, const float * op1, const vec3 & op2) {
    for (unsigned i = 0; i < 3; i++) {
        r[i] = op1[3*i+0] * op2[0] + op1[3*i+1] * op2[1] + op1[3*i+2] * op2[2];
    }
}


bool mul(vec4 & result, const fmatrix & op1, const vec4 & op2) {
    if (op1.cols() == 4) {
        for (unsigned i=0; i<4; i++) {
            result[i] = op1[i][0] * op2[0] + op1[i][1] * op2[1] + op1[i][2] * op2[2] + op1[i][3] * op2[3];
        }
    }
    return true;
}


bool sub(fmatrix & result, const fmatrix & op1, const fmatrix & op2) {
    if (op1.rows() != op2.rows() || op1.cols() != op2.cols()) {
        cout << "Error: sub --> op1 ("<< op1.rows() <<"x"<< op1.cols();
        cout <<") and op2 ("<< op2.rows() <<"x"<< op2.rows() <<")";
        cout << " cannot be substracted: different row or column number\n";
        return false;
    }
    if (result.rows() != op1.rows() || result.cols() != op2.cols()) {
        result.resize(op1.rows(), op1.cols());
    }
    //Performing the Sum.
    float * a = (float *) op1.data();
    float * b = (float *) op2.data();
    float * c = result.data();

    for (unsigned i=0; i < op1.rows() * op1.cols(); i++) {
        *c++ = *a++ - *b++;
    }
    return true;
}




bool block_add(unsigned block_rows, unsigned block_cols,
               fmatrix & dest, unsigned dest_offset_row, unsigned dest_offset_col,
               const fmatrix & orig, unsigned orig_offset_row, unsigned orig_offset_col) {

    if (orig.rows() < orig_offset_row + block_rows ||
        orig.cols() < orig_offset_col + block_cols ||
        dest.rows() < dest_offset_row + block_rows ||
        dest.cols() < dest_offset_col + block_cols) {
        cout << "ERROR: block_add --> The matrices need to be bigger given the offsets.\n";
        cout << "       block size (" << block_rows << "x" << block_cols << ")\n";
        cout << "       orig ("<< orig.rows() << "x"<< orig.cols();
        cout << ") offset_row: " << orig_offset_row << "  offset_col: "<< orig_offset_col << endl;
        cout << "       dest ("<< dest.rows() << "x"<< dest.cols();
        cout << ") offset_row: " << dest_offset_row << "  offset_col: "<< dest_offset_col << endl;
        return false;
    }

    for (unsigned i=0; i<block_rows; i++) {
        for (unsigned j=0; j<block_cols; j++) {
            dest[dest_offset_row+i][dest_offset_col+j] += orig[orig_offset_row+i][orig_offset_col+j];
        }
    }
    return true;
}



bool block_add(fmatrix & dest, unsigned dest_offset_row, unsigned dest_offset_col,
               const fmatrix & orig) {

    return block_add(orig.rows(), orig.cols(),
                     dest, dest_offset_row, dest_offset_col,
                     orig, 0, 0);
}



bool block_set(unsigned block_rows, unsigned block_cols,
               fmatrix & dest, unsigned dest_offset_row, unsigned dest_offset_col,
               const fmatrix & orig, unsigned orig_offset_row, unsigned orig_offset_col) {

    if (orig.rows() < orig_offset_row + block_rows ||
        orig.cols() < orig_offset_col + block_cols ||
        dest.rows() < dest_offset_row + block_rows ||
        dest.cols() < dest_offset_col + block_cols) {
        cout << "ERROR: block_set --> The matrices need to be bigger given the offsets.\n";
        cout << "       block size (" << block_rows << "x" << block_cols << ")\n";
        cout << "       orig ("<< orig.rows() << "x"<< orig.cols();
        cout << ") offset_row: " << orig_offset_row << "  offset_col: "<< orig_offset_col << endl;
        cout << "       dest ("<< dest.rows() << "x"<< dest.cols();
        cout << ") offset_row: " << dest_offset_row << "  offset_col: "<< dest_offset_col << endl;
        return false;
    }

    for (unsigned i=0; i<block_rows; i++) {
        for (unsigned j=0; j<block_cols; j++) {
            dest[dest_offset_row+i][dest_offset_col+j] = orig[orig_offset_row+i][orig_offset_col+j];
        }
    }
    return true;
}




bool block_set(fmatrix & dest, unsigned dest_offset_row, unsigned dest_offset_col,
               const fmatrix & orig) {
    return block_set(orig.rows(), orig.cols(),
                     dest, dest_offset_row, dest_offset_col,
                     orig, 0, 0);
}



void apply_preconditioner(fmatrix & A, fmatrix & b) {
    fmatrix c(A.rows(), A.cols());
    //Applying preconditioner.
    //c.resize(A.rows(), A.cols());
    c.set(0.0);
    for (unsigned i=0; i<A.cols(); i++) {
        if (A[i][i] < -0.000001 || A[i][i] > 0.000001) {
            c[i][i] = 1.0/A[i][i];
        }
    }
    fmatrix temp;
    mul(temp, c, A);
    A.set(temp);

    mul(temp, c, b);
    b.set(temp);
}


///Solve a linear system of ecuations, using Gauss-Seidel method.
///http://es.wikipedia.org/wiki/M%C3%A9todo_de_Gauss-Seidel

///WARNING: This method only works when the diagonal of A is bigger than the
///         rest of values. Otherwise never converges.

bool solve_gs(const fmatrix & A, fmatrix & x, const fmatrix & b) {

    if (b.cols() != 1 || A.rows() != b.rows() || A.rows() != A.cols()) {
        cout << "Error: solve --> A is "<< A.rows() <<"x"<< A.cols();
        cout <<" and b is "<< b.rows() <<"x"<< b.cols() << " INCORRECT\n";
        return false;
    }

    if (!A.is_defined_diag()) {
        cout << "Error: solve --> A has zeroes in the main diag.\n";
        return false;
    }

    if (x.cols() != 1 || A.cols() != x.rows()) {
        x.resize(A.cols(), 1);
    }

    //Initial approximation.
    x.set(1.0);
    //Matrix to compute if the solution is near the convergence.
    fmatrix prev_x(x);

    fmatrix distance;
    unsigned iterations = 0;

    //Starting Gauss-Seidel Method.
    bool convergence = false;
    float temp;

    while (!convergence) {
        //Gauss-Seidel.
        for (unsigned i=0; i<A.rows(); i++) {
            temp = 0.0;
            for (unsigned j=0; j<A.cols(); j++) {
                if (j != i) {
                    temp += A[i][j] * x[j][0];
                }
            }
            x[i][0] = (b[i][0] - temp) / A[i][i];
        }

        //Testing convergence.
        iterations++;
        sub(distance, x, prev_x);

        prev_x.set(x);
        if (distance.abs_summatory() < 0.0001) {
            convergence = true;
        }
    }
    cout << "Iterations " << iterations << endl;
    return true;
}



///Solve the linear system Ax = b using Conjugate Gradient
bool solve_gc(const fmatrix & A, fmatrix & x, const fmatrix & b) {

    if (b.cols() != 1 || A.rows() != b.rows() || A.rows() != A.cols()) {
        cout << "Error: solve --> A is "<< A.rows() <<"x"<< A.cols();
        cout <<" and b is "<< b.rows() <<"x"<< b.cols() << " INCORRECT\n";
        return false;
    }
    //Initial approximation of x.
    x.set(0.0);

    fmatrix c, r, v, alpha, beta, temp1, temp2, temp3, temp4, temp5, temp6, v_t, Av, v_tAv;
    //Computing the residue.
    //mul(temp1, A, x);
    //sub(r, b, temp1);
    //v.set(r);
    r.set(b);
    v.set(b);

    unsigned iterations = 0;

    //When the residue is small enough, the (approximate) solution is found.
    while (iterations < 60 && r.abs_summatory() > 0.00001) {
        mul(Av, A, v);
        //Computing Alpha
        v_t.set(v);
        v_t.transpose();

        mul(v_tAv, v_t, Av);
        if (!v_tAv.invert()) {
            iterations++;
            continue;
        }

        mul(temp1, v_t, r);
        mul(alpha, temp1, v_tAv);

        //Computing next X.
        mul(temp2, alpha, v);
        add(x, x, temp2);

        //Computing next residue.
        mul(temp3, alpha, Av);
        sub(r, r, temp3);

        //Computing Beta.
        mul(temp4, v_t, A);
        mul(temp5, temp4, r);
        mul(beta, temp5, v_tAv);
        beta.negate();

        //Computing next v.
        mul(temp6, beta, v);
        add(v, r, temp6);
        iterations++;
    }
    cout << "Iterations " << iterations << endl;
    return true;
}


bool solve_choleski(const fmatrix & A, fmatrix & x, const fmatrix & b) {
    cout << "\n\nERROR: solve_choleski --> Not implemented yet!!\n\n";
    return false;
}


bool solve_choleski(const fmatrix & A, const fmatrix & A_t, fmatrix & x, const fmatrix & b) {
    cout << "\n\nERROR: solve_choleski --> Not implemented yet!!\n\n";
    return false;
}


bool solve_choleski(const fmatrix & A, const fmatrix & d, const fmatrix & A_t, fmatrix & x, const fmatrix & b) {
    cout << "\n\nERROR: solve_choleski --> Not implemented yet!!\n\n";
    return false;
}
