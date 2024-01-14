#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>

#include <iostream>
#include <fstream>

#include "vec3.h"
#include "material.h"
#include "shape.h"

bool debug = false;
std::string filename = "image";
std::ofstream output_file;

thrust::minstd_rand rng;
thrust::uniform_real_distribution<float> random(0.0f, 1.0f);

__host__ __device__ inline float random_float() {
	return random(rng);
}

__host__ __device__ inline float random_float(float min, float max) {
	return min + (max - min) * random_float();
}