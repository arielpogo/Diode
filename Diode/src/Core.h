#pragma once
//CUDA CORE
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

//CUDA EXTRA
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>
#include "nppdefs.h" //cuda constant maximums

//C, C++
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cmath>

//HOST GLOBALS
bool debug = false;
std::string filename = "image";
std::ofstream output_file;

//DEVICE CONSTS
__constant__ float infinity = NPP_MAXABS_32F; //32 bit float maximum
__constant__ float PI = 3.1415926535897932385f;
#define NUM_OBJECTS 3

//DEVICE HELPERS
__device__ inline double degrees_to_radians(double degrees) {
	return degrees * PI / 180.0;
}