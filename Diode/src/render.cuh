#pragma once
#include "core.h"
#include "ds/vec3.h"
#include "ds/color255.h"
#include "ds/ray.h"
#include "ds/object.h"
#include "ds/interval.h"
#include "ds/material.h"
#include "camera.h"


__device__ bool lambertian_scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered);
__device__ bool metal_scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered);
__device__ bool solid_scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered);

//DEVICE GLOBALS
__device__ camera* d_cam;
__device__ object* d_global_objects;

__global__ void render_kernel(color255* d_result);
__device__ vec3 ray_color(const ray& r, int depth);
__device__ ray get_ray(int i, int j);
__device__ vec3 pixel_sample_square();

__global__ void render_kernel(color255* d_result) {
	int pixel = threadIdx.x + (blockDim.x * blockIdx.x);
	int i = pixel % d_cam->image_width;
	int j = pixel / d_cam->image_width;

	vec3 pixel_color(0,0,0);

	for (int sample = 0; sample < d_cam->samples_per_pixel; sample++) {
		ray r = get_ray(i, j);
		pixel_color += ray_color(r, d_cam->max_bounces);
	}
	
	d_result[pixel].setcolor(pixel_color, d_cam->samples_per_pixel);
}

__device__ vec3 ray_color(const ray& r, int depth) {
	if (depth <= 0) return vec3(0, 0, 0); //limit recursion depth with max bounces  

	hit_record rec;
	bool hit;
	interval ray_t(0.001, infinity);

	float closest_yet = ray_t.max;

	for (int i = 0; i < NUM_OBJECTS; i++) {
		hit_record temp_rec;
		if (d_global_objects[i].hit(r, interval(ray_t.min, closest_yet), temp_rec)) {
			//hit = true;
			//closest_yet = temp_rec.t;
			//rec = temp_rec;
		}
	}

	if (hit) {
		ray scattered;
		vec3 attenuation;
		bool scatter_result = false;
		
		switch (rec.material) {
		case material::lamertian:
			scatter_result = lambertian_scatter(r, rec, attenuation, scattered);
			break;
		case material::metal:
			scatter_result = metal_scatter(r, rec, attenuation, scattered);
			break;
		case material::solid:
			scatter_result = solid_scatter(r, rec, attenuation, scattered);
			break;
		}

		if(scatter_result) return attenuation * ray_color(scattered, depth - 1);
		else return vec3(0, 0, 0);
	}
	
	//vec3 unit_direction = unit_vector(r.direction());
	//auto a = 0.5*(unit_direction.y() + 1.0);
	//return (1.0-a) * color (1,1,1) + a*color(0.5,0.5,0.9);
	return vec3(1, 1, 1);

}

__device__ ray get_ray(int i, int j) {
	//get a randomly sampled camera ray for the given pixel
	vec3 pixel_center = d_cam->pixel00_loc + (i * d_cam->pixel_delta_u) + (j * d_cam->pixel_delta_v);
	vec3 pixel_sample = pixel_center + pixel_sample_square();

	return ray(d_cam->camera_center, pixel_sample - d_cam->camera_center);
}

__device__ vec3 pixel_sample_square() {
	auto x = -0.5 + random_float();
	auto y = -0.5 + random_float();
	return (x * d_cam->pixel_delta_u) + (y * d_cam->pixel_delta_v);
}

__device__ bool lambertian_scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered) {
	vec3 scatter_direction = rec.normal + random_unit_vector(); //this is what makes this lambertian

	if (scatter_direction.near_zero()) scatter_direction = rec.normal; //if the random unit vector nearly zeroes the reflection

	scattered = ray(rec.p, scatter_direction);
	attenuation = rec.object_hit->albedo;
	return true;
}

__device__ bool metal_scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered) {
	vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
	scattered = ray(rec.p, reflected);
	attenuation = rec.object_hit->albedo;
	return true;
}

__device__ bool solid_scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered) {
	attenuation = rec.object_hit->albedo;
	return true;
}