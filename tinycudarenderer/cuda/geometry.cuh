#pragma once
#include <cmath>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

template<int n> struct vec {
    float data[n] = {0};
    HD float& operator[](const int i)       { return data[i]; }
    HD float  operator[](const int i) const { return data[i]; }
};

template<int n> HD float operator*(const vec<n>& lhs, const vec<n>& rhs) {
    float ret = 0;
    for (int i=n; i--; ret+=lhs[i]*rhs[i]);
    return ret;
}

template<int n> HD vec<n> operator+(const vec<n>& lhs, const vec<n>& rhs) {
    vec<n> ret = lhs;
    for (int i=n; i--; ret[i]+=rhs[i]);
    return ret;
}

template<int n> HD vec<n> operator-(const vec<n>& lhs, const vec<n>& rhs) {
    vec<n> ret = lhs;
    for (int i=n; i--; ret[i]-=rhs[i]);
    return ret;
}

template<int n> HD vec<n> operator*(const vec<n>& lhs, const float& rhs) {
    vec<n> ret = lhs;
    for (int i=n; i--; ret[i]*=rhs);
    return ret;
}

template<int n> HD vec<n> operator*(const float& lhs, const vec<n> &rhs) {
    return rhs * lhs;
}

template<int n> HD vec<n> operator/(const vec<n>& lhs, const float& rhs) {
    vec<n> ret = lhs;
    for (int i=n; i--; ret[i]/=rhs);
    return ret;
}

template<> struct vec<2> {
    float x = 0, y = 0;
    HD float& operator[](const int i)       { return i ? y : x; }
    HD float  operator[](const int i) const { return i ? y : x; }
};

template<> struct vec<3> {
    float x = 0, y = 0, z = 0;
    HD float& operator[](const int i)       { return i ? (1==i ? y : z) : x; }
    HD float  operator[](const int i) const { return i ? (1==i ? y : z) : x; }
};

template<> struct vec<4> {
    float x = 0, y = 0, z = 0, w = 0;
    HD float& operator[](const int i)       { return i<2 ? (i ? y : x) : (2==i ? z : w); }
    HD float  operator[](const int i) const { return i<2 ? (i ? y : x) : (2==i ? z : w); }
    HD vec<2> xy()  const { return {x, y};    }
    HD vec<3> xyz() const { return {x, y, z}; }
};

typedef vec<2> vec2;
typedef vec<3> vec3;
typedef vec<4> vec4;

template<int n> HD float norm(const vec<n>& v) {
    return sqrtf(v*v);
}

template<int n> HD vec<n> normalized(const vec<n>& v) {
    return v / norm(v);
}

HD inline vec3 cross(const vec3 &v1, const vec3 &v2) {
    return {v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x};
}

template<int n> struct dt;

template<int nrows,int ncols> struct mat {
    vec<ncols> rows[nrows] = {{}};

    HD       vec<ncols>& operator[] (const int idx)       { return rows[idx]; }
    HD const vec<ncols>& operator[] (const int idx) const { return rows[idx]; }

    HD float det() const {
        return dt<ncols>::det(*this);
    }

    HD float cofactor(const int row, const int col) const {
        mat<nrows-1,ncols-1> submatrix;
        for (int i=nrows-1; i--; )
            for (int j=ncols-1;j--; submatrix[i][j]=rows[i+int(i>=row)][j+int(j>=col)]);
        return submatrix.det() * ((row+col)%2 ? -1 : 1);
    }

    HD mat<nrows,ncols> invert_transpose() const {
        mat<nrows,ncols> adjugate_transpose;
        for (int i=nrows; i--; )
            for (int j=ncols; j--; adjugate_transpose[i][j]=cofactor(i,j));
        return adjugate_transpose/(adjugate_transpose[0]*rows[0]);
    }

    HD mat<nrows,ncols> invert() const {
        return invert_transpose().transpose();
    }

    HD mat<ncols,nrows> transpose() const {
        mat<ncols,nrows> ret;
        for (int i=ncols; i--; )
            for (int j=nrows; j--; ret[i][j]=rows[j][i]);
        return ret;
    }
};

template<int nrows,int ncols> HD vec<ncols> operator*(const vec<nrows>& lhs, const mat<nrows,ncols>& rhs) {
    return (mat<1,nrows>{{lhs}}*rhs)[0];
}

template<int nrows,int ncols> HD vec<nrows> operator*(const mat<nrows,ncols>& lhs, const vec<ncols>& rhs) {
    vec<nrows> ret;
    for (int i=nrows; i--; ret[i]=lhs[i]*rhs);
    return ret;
}

template<int R1,int C1,int C2> HD mat<R1,C2> operator*(const mat<R1,C1>& lhs, const mat<C1,C2>& rhs) {
    mat<R1,C2> result;
    for (int i=R1; i--; )
        for (int j=C2; j--; )
            for (int k=C1; k--; result[i][j]+=lhs[i][k]*rhs[k][j]);
    return result;
}

template<int nrows,int ncols> HD mat<nrows,ncols> operator*(const mat<nrows,ncols>& lhs, const float& val) {
    mat<nrows,ncols> result;
    for (int i=nrows; i--; result[i] = lhs[i]*val);
    return result;
}

template<int nrows,int ncols> HD mat<nrows,ncols> operator/(const mat<nrows,ncols>& lhs, const float& val) {
    mat<nrows,ncols> result;
    for (int i=nrows; i--; result[i] = lhs[i]/val);
    return result;
}

template<int nrows,int ncols> HD mat<nrows,ncols> operator+(const mat<nrows,ncols>& lhs, const mat<nrows,ncols>& rhs) {
    mat<nrows,ncols> result;
    for (int i=nrows; i--; )
        for (int j=ncols; j--; result[i][j]=lhs[i][j]+rhs[i][j]);
    return result;
}

template<int nrows,int ncols> HD mat<nrows,ncols> operator-(const mat<nrows,ncols>& lhs, const mat<nrows,ncols>& rhs) {
    mat<nrows,ncols> result;
    for (int i=nrows; i--; )
        for (int j=ncols; j--; result[i][j]=lhs[i][j]-rhs[i][j]);
    return result;
}

template<int n> struct dt {
    HD static float det(const mat<n,n>& src) {
        float ret = 0;
        for (int i=n; i--; ret += src[0][i] * src.cofactor(0,i));
        return ret;
    }
};

template<> struct dt<1> {
    HD static float det(const mat<1,1>& src) {
        return src[0][0];
    }
};
