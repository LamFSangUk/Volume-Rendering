#include "test.cuh"


#include <iostream>
#include <algorithm>

using namespace std;

__device__ int window_level;
__device__ int window_width;

__device__ float getIntensity(int16_t* d_vol, int width, int height, int depth, 
						float x, float y, float z) {
	float val = d_vol[(int)floor(z) * width * height + (int)floor(y)* width + (int)floor(x)];
	

	// TODO: global variables are always zero. Why?
	//printf("%d %d\n", window_level, window_width);
	float window_min = 50 - 350 / 2;
	float window_max = 50 + 350 / 2;

	if (val < window_min) val = window_min;
	if (val > window_max) val = window_max;

	val = (val - window_min) / (float)(window_max - window_min);
	return val;
}

__device__ bool intersect(glm::vec4 origin, glm::vec4 dir,
	glm::vec4* bounds, float &t) {
	glm::vec4 inv_dir(1. / dir[0], 1. / dir[1], 1. / dir[2], 0.);

	double t1 = (bounds[0][0] - origin[0])*inv_dir[0];
	double t2 = (bounds[1][0] - origin[0])*inv_dir[0];

	double tmin = min(t1, t2);
	double tmax = max(t1, t2);

	for (int i = 1; i < 3; ++i) {
		t1 = (bounds[0][i] - origin[i])*inv_dir[i];
		t2 = (bounds[1][i] - origin[i])*inv_dir[i];

		tmin = max(tmin, min(t1, t2));
		tmax = min(tmax, max(t1, t2));
	}

	t = tmin;

	return tmax > max(tmin, 0.0);
}

// Raycast into the volume
__device__ glm::vec4 rayCast(int16_t* d_vol, glm::vec4 origin, glm::vec4 dir, int width, int height, int depth) {
	// Find start and end points.
	glm::vec4 start, end, min_bound, max_bound;
	min_bound = glm::vec4(-width / 2.f, -height / 2.f, -depth / 2.f, 1);
	max_bound = glm::vec4(width / 2.f, height / 2.f, depth / 2.f, 1);

	glm::vec4 bounds[2];
	bounds[0] = min_bound; bounds[1] = max_bound;

	float t;
	bool is_intersect = intersect(origin, dir, bounds, t);

	if (!is_intersect) {
		// RGBA color
		glm::vec4 color(0, 0, 0, 255);
		return color;
	}
	
	float max_val = 0.0f;

	// Sampling from start to end
	glm::vec4 cur = origin + dir * t;

	while (true) {
		// Terminate condition
		if (cur.x > max_bound.x || cur.x < min_bound.x
			|| cur.y > max_bound.y || cur.y < min_bound.y
			|| cur.z > max_bound.z || cur.z < min_bound.z)
			break;

		// Get origin coordinates
		float x = cur.x + width / 2.f;
		float y = cur.y + height / 2.f;
		float z = cur.z + depth / 2.f;

		if (x < 0 || y < 0 || z < 0 || x >= width || y >= height || z >= depth) {
			// Ray go
			cur.x += dir.x;
			cur.y += dir.y;
			cur.z += dir.z;
			continue;
		}

		float val = getIntensity(d_vol, width, height, depth, x, y, z);
		if (max_val < val) max_val = val;

		// Ray go
		cur.x += dir.x;
		cur.y += dir.y;
		cur.z += dir.z;
	}

	// RGBA color
	glm::vec4 color(max_val * 255, 0, 0, 255);
	return color;
}

__global__ void render(int16_t* d_vol, unsigned char *d_screen, int width, int height, int depth,
	glm::vec3 scr_corner, glm::vec3 scr_delta_x, glm::vec3 scr_delta_y, glm::vec4 ray_dir) {

	int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
	int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
	int idx = idx_y * 1000 * 4 + idx_x * 4;	// 1000 means width, 4 means channel

	glm::vec3 cur = scr_corner + (float)idx_x * scr_delta_x + (float)idx_y * scr_delta_y;

	glm::vec4 origin(cur, 1);
	glm::vec4 color = rayCast(d_vol, origin, ray_dir, width, height, depth);

	d_screen[idx + 0] = color.x;
	d_screen[idx + 1] = color.y;
	d_screen[idx + 2] = color.z;
	d_screen[idx + 3] = color.w;
}

int rayCastCuda(vdcm::Volume* vol, glm::vec3 scr_center, 
		glm::vec3 scr_delta_x, glm::vec3 scr_delta_y, int scr_width, int scr_height, unsigned char *h_screen) {
	//float *raw_data = vol->getBuffer();
	std::vector<std::vector<int16_t> > *raw_data = &(vol->m_volume_data);
	printf("Get volume\n");

	int wl = std::get<0>(vol->getDefaultWindowing());
	window_width = std::get<1>(vol->getDefaultWindowing());
	printf("%d %d\n", wl, window_width);

	cudaMemcpyToSymbol(&window_level, &wl, sizeof(int),0, cudaMemcpyHostToDevice);

	int width = vol->getWidth();
	int height = vol->getHeight();
	int depth = vol->getDepth();

	unsigned char *d_screen;
	cudaMalloc((void**)&d_screen, sizeof(unsigned char) * scr_width * scr_height * 4);
	cudaMemcpy(d_screen, h_screen, sizeof(unsigned char) * scr_width * scr_height * 4, cudaMemcpyHostToDevice);

	// Allocate volume data in gpu.
	int16_t *d_vol;
	cudaMalloc((void**)&d_vol, sizeof(int16_t) * width * height * depth);

	int16_t *dst = d_vol;

	for (std::vector<std::vector<int16_t> >::iterator it = raw_data->begin(); it != raw_data->end(); ++it) {
		int16_t *src = &((*it)[0]);
		size_t sz = it->size();

		cudaMemcpy(dst, src, sizeof(int16_t)*sz, cudaMemcpyHostToDevice);
		dst += sz;
	}


	dim3 block_size(16, 16, 1);
	dim3 numBlocks(scr_width / block_size.x, scr_height / block_size.y);

	glm::vec4 volume_center(0, 0, 0, 1);
	glm::vec4 direction;
	glm::vec3 scr_start_corner;		// Start point left bottom

	scr_start_corner = scr_center - (scr_width / 2.f) * scr_delta_x - (scr_height / 2.f) * scr_delta_y;
	// Assume that only consider parallel ray
	direction = volume_center - glm::vec4(scr_center, 1.);
	direction = glm::normalize(direction);

	render << <numBlocks, block_size >> > (d_vol, d_screen, width, height, depth, scr_start_corner, scr_delta_x, scr_delta_y, direction);

	cudaMemcpy(h_screen, d_screen, sizeof(unsigned char) * scr_width * scr_height * 4, cudaMemcpyDeviceToHost);

	cudaFree(d_screen);
	cudaFree(d_vol);

	return true;
}