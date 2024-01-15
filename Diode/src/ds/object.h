#pragma once
#include "..\core.h"
#include "material.h"
#include "interval.h"
#include "ray.h"
#include "vec3.h"

enum shape;
struct object;

struct hit_record {
	point3 p = point3();
	vec3 normal = vec3();
	material material = material::solid;
	float t = 0.0f;
	bool front_face = false;
	object* object_hit = nullptr;

	//NOTE: outward_normal must have unit length
	//sets normal vector
	__device__ void set_face_normal(const ray& r, const vec3& outward_normal) {
		front_face = dot(r.direction(), outward_normal) < 0;
		normal = front_face ? outward_normal : -outward_normal;
	}
};

__device__ static bool sphere_hit(const ray& r, object& o, interval& ray_t, hit_record& rec);

enum shape {
	sphere = 1
};

struct object {
	__device__ object(vec3 _albedo, vec3 _center, material _material, shape _shape, float _radius) {
		albedo = _albedo;
		center = _center;
		material = _material;
		shape = _shape;
		radius = _radius;
	}

	__device__ bool hit(const ray& r, interval& i, hit_record& rec) {
		switch (shape) {
		case shape::sphere:
			return sphere_hit(r, *this, i, rec);
			break;
		}
	}

	vec3 albedo;
	vec3 center;
	material material;
	shape shape;
	float radius; //temporary for spheres
};

__device__ static bool sphere_hit(const ray& r, object& o, interval& ray_t, hit_record& rec) {
	vec3 oc = r.origin() - o.center; //A-C
	float a = r.direction().length_squared(); //B.B
	float half_b = dot(oc, r.direction());
	float c = oc.length_squared() - o.radius * o.radius;

	float discriminant = half_b * half_b - a * c;
	if (discriminant < 0) return false;
	float sqrtf = sqrt(discriminant);

	//nearest t in the accepted range, test +/-
	float root = (-half_b - sqrtf) / a;
	if (!ray_t.surrounds(root)) {
		root = (-half_b + sqrtf) / a;
		if (!ray_t.surrounds(root)) {
			return false; //no solutions in the range
		}
	}

	rec.t = root;
	rec.p = r.at(rec.t);
	vec3 outward_normal = (rec.p - o.center) / o.radius;
	rec.set_face_normal(r, outward_normal);
	rec.material = o.material;
	rec.object_hit = &o;

	return true;
}

