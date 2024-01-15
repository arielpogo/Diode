#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cmath>

#include "ds/vec3.h"
#include "ds/material.h"
#include "ds/object.h"
#include "ds/color255.h"

#include "nppdefs.h"

#define CHKALLOC(pointer, size)\
	if(!pointer){\
		std::cerr << "Fatal error: Could not allocate " << size << " bytes of memory." << std::endl;\
		goto cleanup;\
	}\

bool debug = false;
std::string filename = "image";
std::ofstream output_file;

__constant__ float infinity = NPP_MAXABS_32F; //32 bit float maximum
__constant__ float PI = 3.1415926535897932385f;

__device__ inline double degrees_to_radians(double degrees) {
	return degrees * PI / 180.0;
}