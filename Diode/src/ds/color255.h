#pragma once
#include "core.h"

class color_255 {
public:
	char e[3];

	__host__ __device__ color_255() : e{ 0,0,0 } {

	}

	__host__ __device__ color_255(char r, char g, char b) : e{ r, g, b } {

	}

	__host__ __device__ color_255(vec3& c, int samples) {
		double r = c.x();
		double g = c.y();
		double b = c.z();

		//divide color by number of samples
		double scale = 1.0 / samples;
		r *= scale;
		g *= scale;
		b *= scale;

		r = sqrt(r);
		g = sqrt(g);
		b = sqrt(b);

		float max_intensity = 0.999f;

		r = (r > max_intensity) ? max_intensity : r;
		g = (g > max_intensity) ? max_intensity : g;
		b = (b > max_intensity) ? max_intensity : b;

		e[0] = static_cast<char>(256 * r);
		e[1] = static_cast<char>(256 * g);
		e[2] = static_cast<char>(256 * b);
	}

	__host__ __device__ char x() const { return e[0]; }
	__host__ __device__ char y() const { return e[1]; }
	__host__ __device__ char z() const { return e[2]; }
};