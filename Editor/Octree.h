#pragma once

#include <Eigen/Dense>

class OctreeNode {
public:
	int level;
	int child[8];
	Eigen::Vector4f bounds[2];

	OctreeNode();
	~OctreeNode();
};

class Octree {
public:
	OctreeNode* root;
};
