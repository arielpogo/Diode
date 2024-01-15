#pragma once
#include "core.h"

__device__ thrust::minstd_rand* rng;
__device__ thrust::uniform_real_distribution<float>* random;

__device__ inline float random_float() {
	return (*random)(*rng);
}

__device__ inline float random_float(float min, float max) {
	return min + (max - min) * random_float();
}


using std::sqrt;

class vec3 {
public:
	float e[3];

	//default constructor
	 __host__ __device__ vec3() : e{ 0,0,0 } {

	}

	//parameterized constructor
	 __host__ __device__ vec3(float e0, float e1, float e2) : e{ e0, e1, e2 } {

	}

	//accessors
	 __host__ __device__ float x() const { return e[0]; }
	  __host__ __device__ float y() const { return e[1]; }
	 __host__ __device__ float z() const { return e[2]; }

	//simple operators, negation, access
	 __host__ __device__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
	 __host__ __device__ float operator[](int i) const { return e[i]; }
	 __host__ __device__ float& operator[](int i) { return e[i]; }

	//add other vector's stuff to this and return this as the answer
	 __host__ __device__ vec3& operator+=(const vec3& v) {
		e[0] += v.e[0];
		e[1] += v.e[1];
		e[2] += v.e[2];
		return *this;
	}

	//ditto, but with another float
	 __host__ __device__ vec3& operator*=(float t) {
		e[0] *= t;
		e[1] *= t;
		e[2] *= t;
		return *this;
	}

	//ditto, using fractions
	 __host__ __device__ vec3& operator/=(float t) {
		return *this *= 1 / t;
	}

	 __host__ __device__ float length() const {
		return sqrt(length_squared());
	}

	 __host__ __device__ float length_squared() const {
		return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
	}

	//useful for telling if reflections are nearly negated as to not reflect at all
	 __host__ __device__ bool near_zero() const {
		auto s = 1e-8;
		return(fabs(e[0]) < s && (fabs(e[1]) < s) && (fabs(e[2]) < s));
	}

	 __device__ static vec3 random() {
		return vec3(random_float(), random_float(), random_float());
	}

	 __device__ static vec3 random(float min, float max) {
		return vec3(random_float(min, max), random_float(min, max), random_float(min, max));
	}
};

//point3 is an alias of vec3
using point3 = vec3;

//allow vec3 to be printed, written to a file etc., especially for PPM files
 __host__ __device__ inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
	return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

//add vec3s together
 __host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v) {
	return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

//subtract vec3s
 __host__ __device__ inline vec3 operator-(const vec3& u, const vec3& v) {
	return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

//multiply vec3s
 __host__ __device__ inline vec3 operator*(const vec3& u, const vec3& v) {
	return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

//multiply a float with a vec3
 __host__ __device__ inline vec3 operator*(float t, const vec3& v) {
	return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

//multiply a vec3 with a float
 __host__ __device__ inline vec3 operator*(const vec3& v, float t) {
	return t * v;
}

//divide a vec3 by a float
 __host__ __device__ inline vec3 operator/(vec3 v, float t) {
	return (1 / t) * v;
}

//return the dot product of two vec3s
 __host__ __device__ inline float dot(const vec3& u, const vec3& v) {
	return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

//return the cross product of two vec3s
 __host__ __device__ inline vec3 cross(const vec3& u, const vec3& v) {
	return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
		u.e[2] * v.e[0] - u.e[0] * v.e[2],
		u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

//return the unit vector of a vec3
 __host__ __device__ inline vec3 unit_vector(vec3 v) {
	return v / v.length();
}

//produces a random vec3 which is in a sphere of 1,1,1
  __device__ inline vec3 random_in_unit_sphere() {
	while (true) {
		//generating vectors in a sphere is difficult, the simplist way is to keep generating within a box until they are
		//a length of 1 (within a sphere if placed in the center, longer than 1 is outside a sphere
		//also we can't shrink the box within the sphere or else there would be some direction impossible to generate (circled square vs squared circle)
		vec3 p = vec3::random(-1, 1);
		if (p.length_squared() < 1) return p;
	}
}

 __device__ inline vec3 random_unit_vector() {
	return unit_vector(random_in_unit_sphere());
}

//a random vec3 within the same hemisphere as a given vec3, usually a surface normal
//one application is diffuse materials where we reflect off of the surface in a random
//direction in the same hemisphere as the normal (thus not into the sphere itself)
//we can tell by seeing if dot prod. is positive, if not then flip
 __device__ inline vec3 random_on_hemisphere(const vec3& normal) {
	vec3 on_unit_sphere = random_unit_vector();
	if (dot(on_unit_sphere, normal) > 0.0) return on_unit_sphere;
	else return -on_unit_sphere;
}

 __host__ __device__ vec3 reflect(const vec3& v, const vec3& n) {
	return v - 2 * dot(v, n) * n;
}