#pragma once
#include "core.h"

class camera {
public:
	static const int DEFAULT_HEIGHT = 720;

	//image
	int image_height = DEFAULT_HEIGHT;
	double aspect_ratio = 16.0 / 9.0;
	double hfov = 90; //horizontal fov in degrees

	int samples_per_pixel = 10;
	int max_bounces = 10; //per ray

	point3 lookfrom = point3(0, 0, 0);
	point3 lookat = point3(0, 0, 0);
	vec3 relative_up = vec3(0, 1, 0);

	//get initialized below
	int image_width;
	point3 camera_center;
	point3 pixel00_loc;
	vec3 pixel_delta_u;
	vec3 pixel_delta_v;
	vec3 u, v, w; //camera vectors

	__device__ void initialize() {
		//viewport, camera
		camera_center = lookfrom;
		double focal_length = (lookfrom - lookat).length();

		//split the right triangle that is the view width into two, thus the hfov in two
		//tan(theta) = o/a, o is half of the width
		//so multiplying by a (focal length) gives the viewport width
		double viewport_width = 2 * tan(degrees_to_radians(hfov) / 2) * focal_length;
		double viewport_height = viewport_width / ((double)image_width / (double)image_height);//recalculated because of int rounddown with the width, we want to be closer to what is calculated and not the perfect ratio

		//calculate vectors across the horz and down the vert frame
		relative_up = vec3(0, 1, 0);
		w = unit_vector(lookfrom - lookat);
		u = unit_vector(cross(relative_up, w));
		v = cross(w, u);

		//keep track of the viewport's edges
		//-v because we render from topleft down, while
		//world is down to up
		vec3 viewport_u = viewport_width * u;
		vec3 viewport_v = viewport_height * -v;

		//calculate the distance between pixels in wordspace
		pixel_delta_u = viewport_u / image_width;
		pixel_delta_v = viewport_v / image_height;

		//calculate the location of the topleft pixel
		point3 viewport_upper_left = camera_center - (focal_length * w) - viewport_u / 2 - viewport_v / 2;
		pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
	}
};