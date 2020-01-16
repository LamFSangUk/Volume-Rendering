#pragma once
#include "volume.h"
#include "Octree.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

__global__ void render(float* d_vol, unsigned char *d_screen, int width, int height, int depth,
	glm::vec3 scr_start_corner, glm::vec3 scr_delta_x, glm::vec3 scr_delta_y,
	glm::vec4 ray_dir);

int rayCastCuda(vdcm::Volume* vol,  glm::vec3 scr_center,
	glm::vec3 scr_delta_x, glm::vec3 scr_delta_y, int a, int b, unsigned char *h_screen);

void copyVolCuda(vdcm::Volume* vol);
void copyOctree(Octree *);
void allocateScreenCuda(int, int);