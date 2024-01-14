#pragma once
#include "core.h"

enum shape {
	sphere = 1
};

struct object {
	object(vec3 _albedo, vec3 _location, material _material, shape _shape) {
		albedo = _albedo;
		location = _location;
		material = _material;
		shape = _shape;
	}

	vec3 albedo;
	vec3 location;
	material material;
	shape shape;
};