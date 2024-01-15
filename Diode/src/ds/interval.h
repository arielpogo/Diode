#pragma once
#include "..\core.h"

class interval {
public:
	float min, max;

	__device__ interval() : min(+infinity), max(-infinity) {} //empty on default
	__device__ interval(float _min, float _max) : min(_min), max(_max) {}

	//whether x is in range
	__device__ bool contains(float x) const {
		return min <= x && x <= max;
	}

	//whether x is in range, exclusive
	__device__ bool surrounds(float x) const {
		return min < x && x < max;
	}

	__device__ float clamp(float x) const {
		if (x < min) return min;
		else if (x > max) return max;
		else return x;
	}

	static const interval empty, universe;
};

const static interval empty(+infinity, -infinity);
const static interval universe(-infinity, +infinity);