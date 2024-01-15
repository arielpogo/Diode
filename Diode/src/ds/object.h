#pragma once
#include "core.h"

enum shape {
	sphere = 1
};

struct object {
	__device__ object(vec3 _albedo, vec3 _location, material _material, shape _shape, float _radius) {
		albedo = _albedo;
		location = _location;
		material = _material;
		shape = _shape;
		radius = _radius;
	}

	vec3 albedo;
	vec3 location;
	material material;
	shape shape;
	float radius; //temporary for spheres
};