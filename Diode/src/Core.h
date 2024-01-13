#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

bool debug = false;
std::string filename = "image";
std::ofstream output_file;