#pragma once

#include <vector>
#include <glm/glm.hpp>

typedef std::vector<std::vector<int16_t> > Vol;


//unsigned int LUT[8][3] = 
//{
//	{1, 2, 4},
//	{8, 3, 5},
//	{3, 8, 6},
//	{8, 8, 7},
//	{5, 6, 8},
//	{8, 7, 8},
//	{7, 8, 8},
//	{8, 8, 8}
//};
// 


class OctreeNode {
public:
	int level;
	int child[8];
	glm::vec3 bounds[2];

	int intensity_min;
	int intensity_max;

	OctreeNode();
	//~OctreeNode();
	
};

class OctreeStack {
	int idx;
	glm::vec3 t0;
	glm::vec3 t1;
	int child_num;
};

class Octree {
public:
	OctreeNode* root;

	Octree(Vol* vol, int vol_width, int vol_height, int vol_depth, int max_level);
	int size;
private:
	int m_max_level;
};

void build_octree(Vol* vol, Octree* tree, OctreeNode& cur_node, int, int, int);