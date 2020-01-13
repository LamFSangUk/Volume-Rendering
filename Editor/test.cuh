#pragma once
#include "volume.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

__global__ void sum(float* d_vol, unsigned char *d_screen, int width, int height, int depth,
	float screen_x, float screen_y, float delta_x, float delta_y,
	glm::vec4 ray_dir);

int rayCastCuda(int a, int b, unsigned char *h_screen);