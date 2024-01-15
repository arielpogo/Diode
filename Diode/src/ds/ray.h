#pragma once
#include "..\core.h"

class ray {
public:
	//default constructor
	__device__ ray() {}

	//parameterized constructor
	__device__ ray(const point3& origin, const vec3& direction) : orig(origin), dir(direction) {}

	//accessors
	__device__ point3 origin() const { return orig; }
	__device__ vec3 direction() const { return dir; }

	//a given point along the ray, t in the equation below
	__device__ point3 at(double t) const {
		return orig + t * dir;
	}

private:
	//A ray is represented as
	//A + tB
	point3 orig; //origin, A
	vec3 dir; //direction, B
};
