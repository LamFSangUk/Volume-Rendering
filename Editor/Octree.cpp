#include "Octree.h"

#include <cmath>

OctreeNode::OctreeNode() {

}

void build_octree(Vol* vol, Octree* tree, OctreeNode& cur_node, int cur_idx, int cur_level, int max_level) {
	//printf("%d %d\n", cur_level, cur_idx);

	if (cur_level == max_level) { // Leaf node
		// Get min and max intensity on volume data

		int max_i = INT_MIN;
		int min_i = INT_MAX;
		//printf("From: %f %f %f\n", cur_node->bounds[0].x, cur_node->bounds[0].y, cur_node->bounds[0].z);
		//printf("To: %f %f %f\n", cur_node->bounds[1].x, cur_node->bounds[1].y, cur_node->bounds[1].z);

		for (int z = cur_node.bounds[0].z; z < cur_node.bounds[1].z; z++) {
			for (int y = cur_node.bounds[0].y; y < cur_node.bounds[1].y; y++) {
				for (int x = cur_node.bounds[0].x; x < cur_node.bounds[1].x; x++) {
					int16_t intensity = (*vol)[z][y* tree->root[0].bounds[1].x + x];

					if (intensity > max_i) max_i = intensity;
					if (intensity < min_i) min_i = intensity;
				}
			}
		}
		cur_node.intensity_max = max_i;
		cur_node.intensity_min = min_i;

		return;
	}

	int width = cur_node.bounds[1].x - cur_node.bounds[0].x;
	int height = cur_node.bounds[1].y - cur_node.bounds[0].y;
	int depth = cur_node.bounds[1].z - cur_node.bounds[0].z;

	int max_i = INT_MIN;
	int min_i = INT_MAX;
	for (int i = 0; i < 8; i++) {
		int child_idx = cur_idx * 8 + i + 1;
		cur_node.child[i] = child_idx;

		OctreeNode& child = (tree->root[cur_node.child[i]]);

		child.level = cur_level + 1;
		// Set bounds
		child.bounds[0] = cur_node.bounds[0];
		child.bounds[1] = child.bounds[0] + (cur_node.bounds[1] - cur_node.bounds[0]) / 2.f;
		if (i & 4) {
			child.bounds[0] += glm::vec3(0, 0, depth / 2.f);
			child.bounds[1] += glm::vec3(0, 0, depth / 2.f);
		}
		if (i & 2) {
			child.bounds[0] += glm::vec3(0, height / 2.f, 0);
			child.bounds[1] += glm::vec3(0, height / 2.f, 0);
		}
		if (i % 2) {
			child.bounds[0] += glm::vec3(width / 2.f, 0, 0);
			child.bounds[1] += glm::vec3(width / 2.f, 0, 0);
		}

		build_octree(vol, tree, child, child_idx, cur_level + 1, max_level);
		if (child.intensity_max > max_i) max_i = child.intensity_max;
		if (child.intensity_min < min_i) min_i = child.intensity_min;

	}
	cur_node.intensity_max = max_i;
	cur_node.intensity_min = min_i;
}

Octree::Octree(Vol* vol, int vol_width, int vol_height, int vol_depth, int max_level) {
	// Set Octree max_level

	int total_num_node = 0;
	for (int i = 0; i <= max_level; i++) {
		total_num_node += pow(8, i);
	}

	// Create OctreeNodes
	root = new OctreeNode[total_num_node+10];
	printf("total:%d\n", total_num_node);
	
	this->size = total_num_node;
	// Recusive build botton-up
	OctreeNode& cur_node = root[0];
	int cur_level = 0;
	
	cur_node.level = cur_level;
	cur_node.bounds[0] = glm::vec3(0, 0, 0);
	cur_node.bounds[1] = glm::vec3(vol_width, vol_height, vol_depth);

	build_octree(vol, (Octree*)this, cur_node, 0, cur_level, max_level);
}
