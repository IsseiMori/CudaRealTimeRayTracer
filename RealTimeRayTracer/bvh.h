#ifndef BVH
#define BVH

#include <math.h>
#include <stdlib.h>
#include <iostream>

#include "vec3.h"
#include "hitable.h"
#include "ray.h"
#include "aabb.h"

struct morton_info {
	int hitable_index;		// index of object in original unordered list
	unsigned int morton_code;	// x0y0z0x1y1z1...

	__device__ bool operator < (const morton_info& m) const {
		return morton_code < m.morton_code;
	}

	__device__ void swap(morton_info* m) {
		int h = hitable_index;
		unsigned int mo = morton_code;
		hitable_index = m->hitable_index;
		morton_code = m->morton_code;
		m->hitable_index = h;
		m->morton_code = mo;
	}
};

// Temporary object used to create bvh tree
struct bvh_info {
	size_t hitable_index;	// index to unorded list?
	aabb box;
	vec3 centroid;
};


class bvh_node : public hitable {
public:
	__device__ bvh_node() {}
	
	__device__ bvh_node(hitable** list, int list_size, float time0, float time1);

	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
	__device__ virtual bool bounding_box(float t0, float t1, aabb& output_box) const;

public:
	bvh_node* left;		// Left child
	bvh_node* right;	// Right child
	aabb box;			// BB for all children
};

class bvh : public hitable {
public:
	__device__ bvh() {}
	__device__ bvh(hitable** l, int l_size);
public:
	hitable** hitable_list;	// list of hitable objects sorted based on morton code
	bvh_node* nodes = nullptr;	// head pointer of bvh tree? will address later...
};

__device__ bvh::bvh(hitable** l, int l_size) {

}

__device__ bool bvh_node::bounding_box(float t0, float t1, aabb& output_box) const {
	output_box = aabb(vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 0.0f));
	return true;
}



__device__ bool bvh_node::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
	if (!box.hit(r, t_min, t_max))
		return false;

	bool hit_left = left->hit(r, t_min, t_max, rec);
	bool hit_right = right->hit(r, t_min, t_max, rec);

	return hit_left || hit_right;
}

__device__ bvh_node::bvh_node(hitable** list, int list_size, float time0, float time1) {
	
}

// Shift bits of morton encode
__device__ inline unsigned int expand_bits(unsigned int v) {
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}

__device__ inline unsigned int encode_morton3(const vec3& v) {
	float x = fmin(fmax(v[0] * 1024.0f, 0.0f), 1023.0f);
	float y = fmin(fmax(v[1] * 1024.0f, 0.0f), 1023.0f);
	float z = fmin(fmax(v[2] * 1024.0f, 0.0f), 1023.0f);
	unsigned int xx = expand_bits((unsigned int)x);
	unsigned int yy = expand_bits((unsigned int)y);
	unsigned int zz = expand_bits((unsigned int)z);
	return xx * 4 + yy * 2 + zz;
}

__device__ inline unsigned int LeftShift3(unsigned int x) {
	if (x == (1 << 10)) --x;

	x = (x | (x << 16)) & 0b00000011000000000000000011111111;
	// x = ---- --98 ---- ---- ---- ---- 7654 3210
	x = (x | (x << 8)) & 0b00000011000000001111000000001111;
	// x = ---- --98 ---- ---- 7654 ---- ---- 3210
	x = (x | (x << 4)) & 0b00000011000011000011000011000011;
	// x = ---- --98 ---- 76-- --54 ---- 32-- --10
	x = (x | (x << 2)) & 0b00001001001001001001001001001001;
	// x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0

	//x = (x | (x << 16)) & 0x30000ff;
	//// x = ---- --98 ---- ---- ---- ---- 7654 3210
	//x = (x | (x << 8)) & 0x300f00f;
	//// x = ---- --98 ---- ---- 7654 ---- ---- 3210
	//x = (x | (x << 4)) & 0x30c30c3;
	//// x = ---- --98 ---- 76-- --54 ---- 32-- --10
	//x = (x | (x << 2)) & 0x9249249;
	//// x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0

	return x;
}

__device__ inline unsigned int EncodeMorton3(const vec3& v) {
	return (LeftShift3(v[2]) << 2) | (LeftShift3(v[1]) << 1) | LeftShift3(v[0]);
}


#endif